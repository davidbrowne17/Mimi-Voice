# Mimi-Voice
 Create Unmute voice embeddings

### Setup

```bash
git clone git@github.com:davidbrowne17/mimi-voice.git
cd Mimi-Voice
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create the voice embeddings
python create_voice.py <input_folder> <output_folder>
# You will need access to huggingface
huggingface-cli login
```