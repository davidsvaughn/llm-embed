# llm-embed
Turn LLMs into embedding models. A framework for fine-tuning LLMs into high-performance embedding models using a novel hybrid objective function that combines siamese contrastive loss with causal language modeling loss. 

### setup
```
git clone https://github.com/davidsvaughn/llm-embed
cd llm-embed
virtualenv -p python3.10 venv && source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install flash-attn --no-build-isolation
# pip install -e .
```

### install exllamav2
```
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt
pip install -e .
cd ..
```

### install SentenceTransformers
```
git clone https://github.com/UKPLab/sentence-transformers
cd sentence-transformers
python -m pip install -e ".[dev]"
pre-commit install
cd ..
```

### HuggingFace login
```
huggingface-cli login --token $HUG_READ_TOKEN
huggingface-cli login --token $HUG_WRITE_TOKEN
```

### wandb login
run `wandb login --cloud`

### pre-process data
- put `[bw|fw|math]_[training|validation].json` files in `data/[bw|fw|math]` sub-directories
- run `python preprocess_datasets.py`

### generate pairwise training dataset
- generate predictions with weak model
- generate pairwise dataset using predictions to estimate **hard pairs** (i.e. **hard negatives/positives**)
1. download weak model
2. generate predictions
```
torchrun --nproc_per_node 4 siamese_test.py \
    --item-type bw \
    gen \
    --model-dir ~/models \
    --model-id dan-bw
```

### fine-tune embedding model
```
# single GPU
python siamese_train.py

# multi-GPU
torchrun --nproc_per_node 4 siamese_train.py
```

### test embedding model
```
# single GPU
python siamese_test.py \
    --item-type math \
    scan \
    --model-dir output3 \
    --items 123362,33082,13272,27218,29632,31600,52414,78382

# multi-GPU
torchrun --nproc_per_node 4 siamese_test.py \
    --item-type math \
    scan \
    --model-dir output3 \
    --items 123362,33082,13272,27218,29632,31600,52414,78382
```

### merge adapter model
```
python merge_adapter.py --checkpoint_dir output/checkpoint-2400
```

### upload merged model
```
huggingface-cli upload davidsvaughn/phi4-math-lasttoken-1 output/model --private
```