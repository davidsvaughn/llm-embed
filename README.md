# llm-embed
Turn LLMs into embedding models. A framework for fine-tuning LLMs into high-performance embedding models using a novel hybrid objective function that combines siamese contrastive loss with causal language modeling loss. 

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

HF login
```
huggingface-cli login --token $HUG_READ_TOKEN
huggingface-cli login --token $HUG_WRITE_TOKEN
```

wandb login
```
wandb login --cloud
```

fine-tune embedding model
```
# single GPU
python siamese_train.py

# multi-GPU
torchrun --nproc_per_node 4 siamese_train.py
```

test embedding model
```
# single GPU
python siamese_test.py

# multi-GPU
torchrun --nproc_per_node 4 siamese_test.py
```

merge adapter model
```
python merge_adapter.py --checkpoint_dir output/checkpoint-2400
```

upload merged model
```
huggingface-cli upload davidsvaughn/phi4-math-lasttoken-1 output/model --private
```