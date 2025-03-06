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
git clone https://github.com/turboderp-org/exllamav2
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
1. download weak model:
```
huggingface-cli download davidsvaughn/dan-bw --local-dir ~/models/dan-bw
```

2. generate weak predictions:
```
torchrun --nproc_per_node 4 siamese_test.py \
    --item-type bw \
    --pooling-mode mean \
    gen \
    --model-dir ~/models \
    --model-id dan-bw \
    --item-filter n%2!=0 # odd item numbers only

# or use bash script
bash scripts/gen_preds.sh
```

3. generate pairwise dataset using predictions (from step 2) to estimate *hard pairs* (i.e. *hard negatives/positives*)
```
# command here...
```

### fine-tune embedding model
```
# single GPU
python siamese_train.py

# multi-GPU
torchrun --nproc_per_node 4 siamese_train.py

# or bash script
bash scripts/finetune.sh
```

### test embedding model / scan checkpoints to find optimum
```
torchrun --nproc_per_node 4 siamese_test.py \
    --item-type bw \
    --pooling-mode lasttoken \
    scan \
    --model-dir output6 \
    --chk-min 200 --chk-max 1500 \
    --items 33234,63166,96340,58566,95508,104462,34002,96326,63172,126288

# or bash script
bash scripts/scan_chkpts.sh
```

### merge adapter model
```
python merge_adapter.py --checkpoint_dir output/checkpoint-2400 --model_dir ~/models/new-model
```

### upload merged model
```
huggingface-cli upload davidsvaughn/phi4-math ~/models/new-model --private
```

### exl2 quantize merged model (e.g. to 4 bit)
```
bash scripts/exl2_convert.sh ~/models/new-model 4 
```