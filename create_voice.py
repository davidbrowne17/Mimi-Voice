import math
import sys
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import julius
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download
from moshi.models import loaders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
APPLY_CLEANING = True

class MimiAdapter(nn.Module):
    """Adapter network to transform base Mimi embeddings to mimi_voice space"""
    def __init__(self, dim=512, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.1)
            ))
        self.final_norm = nn.LayerNorm(dim)
        self.final_proj = nn.Linear(dim, dim)
        with torch.no_grad():
            self.final_proj.weight.data = torch.eye(dim) + torch.randn(dim, dim) * 0.02
            self.final_proj.bias.data.zero_()
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        # x shape: [batch, dim, time]
        residual = x
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = x + layer(x)
        x = self.final_norm(x)
        x = self.final_proj(x)
        x = x.transpose(1, 2)  # [batch, dim, time]
        x = x + residual
        x = x * self.scale
        return x

def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 22, energy_floor: float = 2e-3):
    wav = wav - wav.mean(dim=-1, keepdim=True)
    energy = wav.std()
    if energy < energy_floor: return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    try: input_loudness_db = transform(wav).item()
    except RuntimeError: return wav
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    assert output.isfinite().all()
    return output

def sinc(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t == 0, torch.ones(1, device=t.device, dtype=t.dtype), torch.sin(t) / t)

def kernel_upsample2(zeros=56, device=None):
    win = torch.hann_window(4 * zeros + 1, periodic=False, device=device)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros, device=device)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel

def upsample2(x, zeros=56):
    *other, time = x.shape
    kernel = kernel_upsample2(zeros, x.device).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)

def kernel_downsample2(zeros=56, device=None):
    win = torch.hann_window(4 * zeros + 1, periodic=False, device=device)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros, device=device)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel

def downsample2(x, zeros=56):
    if x.shape[-1] % 2 != 0: x = F.pad(x, (0, 1))
    xeven, xodd = x[..., ::2], x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros, x.device).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(*other, time)
    return out.view(*other, -1).mul(0.5)

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim) if bi else None
    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear: x = self.linear(x)
        return x, hidden

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None: conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)): rescale_conv(sub, reference)

class Demucs(nn.Module):
    def __init__(self, chin=1, chout=1, hidden=48, depth=5, kernel_size=8, stride=4, causal=True, resample=4, growth=2, max_hidden=10_000, normalize=True, glu=True, rescale=0.1, floor=1e-3, sample_rate=16_000):
        super().__init__()
        if resample not in [1, 2, 4]: raise ValueError("Resample should be 1, 2 or 4.")
        self.chin, self.chout, self.hidden, self.depth, self.kernel_size, self.stride, self.causal, self.floor, self.resample, self.normalize, self.sample_rate = chin, chout, hidden, depth, kernel_size, stride, causal, floor, resample, normalize, sample_rate
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        activation, ch_scale = (nn.GLU(1), 2) if glu else (nn.ReLU(), 1)
        for index in range(depth):
            encode = [nn.Conv1d(chin, hidden, kernel_size, stride), nn.ReLU(), nn.Conv1d(hidden, hidden * ch_scale, 1), activation]
            self.encoder.append(nn.Sequential(*encode))
            decode = [nn.Conv1d(hidden, ch_scale * hidden, 1), activation, nn.ConvTranspose1d(hidden, chout, kernel_size, stride)]
            if index > 0: decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout, chin, hidden = hidden, hidden, min(int(growth * hidden), max_hidden)
        self.lstm = BLSTM(chin, bi=not causal)
        if rescale: rescale_module(self, reference=rescale)
    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for _ in range(self.depth): length = math.ceil((length - self.kernel_size) / self.stride) + 1
        for _ in range(self.depth): length = (length - 1) * self.stride + self.kernel_size
        return int(math.ceil(length / self.resample))
    def forward(self, mix):
        if mix.dim() == 2: mix = mix.unsqueeze(1)
        if self.normalize: std = mix.std(dim=-1, keepdim=True); mix = mix / (self.floor + std)
        else: std = 1
        length, x = mix.shape[-1], mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample > 1: x = upsample2(x)
        if self.resample > 2: x = upsample2(x)
        skips = []
        for encode in self.encoder: x = encode(x); skips.append(x)
        x, _ = self.lstm(x.permute(2, 0, 1)); x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1); x = x + skip[..., :x.shape[-1]]; x = decode(x)
        if self.resample > 2: x = downsample2(x)
        if self.resample > 1: x = downsample2(x)
        return std * x[..., :length]

def get_demucs():
    model = Demucs(hidden=64)
    state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th", map_location='cpu')
    model.load_state_dict(state_dict)
    return model

class Cleaner(nn.Module):
    def __init__(self, dry_fraction: float = 0.02, sample_rate: int = 24000):
        super().__init__()
        self.dry_fraction, self.sample_rate = dry_fraction, sample_rate
        self._demucs = get_demucs()
        demucs_sr = self._demucs.sample_rate
        self._lowpass = julius.lowpass.LowPassFilter(demucs_sr / sample_rate / 2)
        self._downsample = julius.resample.ResampleFrac(sample_rate, demucs_sr)
        self._upsample = julius.resample.ResampleFrac(demucs_sr, sample_rate)
    @torch.no_grad()
    def forward(self, wav: torch.Tensor):
        low, high = self._lowpass(wav), wav - self._lowpass(wav)
        low = self._downsample(low, full=True)
        denoised = self._demucs(low)
        denoised = (1 - self.dry_fraction) * denoised + self.dry_fraction * low
        denoised = self._upsample(denoised, output_length=wav.shape[-1])
        denoised = denoised + high
        return normalize_loudness(denoised, self.sample_rate)

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process WAV files with Mimi model')
    parser.add_argument('input_folder', type=str, help='Path to input folder containing WAV files')
    parser.add_argument('output_folder', type=str, help='Path to output folder for safetensors')
    args = parser.parse_args()

    output_path = Path(args.output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load the public Mimi model
    print("Loading base Mimi model...")
    mimi_weight_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        loaders.DEFAULT_REPO, mimi_weights=mimi_weight_path,
    )
    checkpoint_info.lm_config = None
    mimi = checkpoint_info.get_mimi(device=DEVICE)
    mimi.set_num_codebooks(16)
    mimi.eval()

    # Load the finetuned adapter model from HuggingFace
    print("Loading finetuned adapter from DavidBrowne17/Mimi-Voice...")
    adapter_model_path = hf_hub_download("DavidBrowne17/Mimi-Voice", "mimi-voice.safetensors")
    adapter = MimiAdapter().to(DEVICE)
    try:
        # Load from safetensors format
        state_dict = load_file(adapter_model_path)
        
        # The safetensors file contains keys like "adapter_state_dict.layers.0.0.weight"
        # We need to remove the "adapter_state_dict." prefix
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("adapter_state_dict."):
                new_key = key[len("adapter_state_dict."):]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        adapter.load_state_dict(cleaned_state_dict)
        adapter.eval()
        print(f"Successfully loaded adapter with {len(cleaned_state_dict)} parameters")
    except FileNotFoundError:
        print(f"FATAL: Adapter model not found at HuggingFace repository.", file=sys.stderr)
        print("Please check that DavidBrowne17/Mimi-Voice contains mimi-voice.safetensors.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Error loading safetensors file: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Initialize the Cleaner
    cleaner = None
    if APPLY_CLEANING:
        print("Initializing Cleaner...")
        cleaner = Cleaner(sample_rate=mimi.sample_rate)
        cleaner.to(device=DEVICE)
        cleaner.eval()

    wav_files = list(Path(args.input_folder).glob("**/*.wav"))
    if not wav_files:
        print(f"No WAV files found in {args.input_folder}")
        return

    # Main processing loop
    for file in wav_files:
        print(f"\nProcessing {file.name}...")

        # STEP 1: Load and process audio
        waveform, sr = torchaudio.load(file)
        if sr != mimi.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, mimi.sample_rate)(waveform)

        length = int(mimi.sample_rate * 10.0)
        wav = waveform[:, :length].float()
        wav = wav.mean(dim=0, keepdim=True)[None]

        # Apply the cleaner, as required.
        if cleaner is not None:
            wav_cleaned = cleaner(wav.to(DEVICE)).clamp(-0.99, 0.99)
        else:
            wav_cleaned = wav.to(DEVICE)

        missing = length - wav_cleaned.shape[-1]
        if missing > 0:
            wav_cleaned = torch.nn.functional.pad(wav_cleaned, (0, missing))

        # STEP 2: Generate the base embedding from the public model
        with torch.no_grad():
            base_emb = mimi.encode_to_latent(wav_cleaned, quantize=False)

        print(f"  - Base (public model) stats: Mean={base_emb.mean():.4f}, Std={base_emb.std():.4f}")

        # STEP 3: Apply the finetuned adapter to transform the embedding
        with torch.no_grad():
            final_emb = adapter(base_emb)

        print(f"  - Final adapted stats:       Mean={final_emb.mean():.4f}, Std={final_emb.std():.4f}")

        # STEP 4: Save the final, usable embedding
        ext = ".1e68beda@240.safetensors"
        out_file = output_path / (file.stem + ext)
        tensors = {"speaker_wavs": final_emb.cpu().contiguous()}
        metadata = {"epoch": "240", "sig": "1e68beda"}
        save_file(tensors, out_file, metadata)
        print(f"  - Saved adapted embedding to: {out_file}")

if __name__ == "__main__":
    main()