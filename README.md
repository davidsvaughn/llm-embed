# llm-embed
turn LLMs into embedding models

setup
```
git clone https://github.com/davidsvaughn/llm-embed
cd llm-embed
virtualenv -p python3.10 venv && source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install flash-attn --no-build-isolation
# pip install -e .
```

HF login
```
huggingface-cli login --token $HUG_READ_TOKEN
huggingface-cli login --token $HUG_WRITE_TOKEN
```

exllamav2
```
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt
pip install -e .
cd ..
```

SentenceTransformers
```
git clone https://github.com/UKPLab/sentence-transformers
cd sentence-transformers
python -m pip install -e ".[dev]"
pre-commit install
cd ..
```


```
wandb login --cloud

python siamese_train.py
torchrun --nproc_per_node 4 siamese_train.py

```