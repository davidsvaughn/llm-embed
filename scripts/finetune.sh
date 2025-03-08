#!/bin/bash

# Fine-tune an embedding model on a pairwise dataset

MODEL_ID="microsoft/Phi-4-mini-instruct" # base model to finetune

# DATASET_ID="davidsvaughn/bw_pairs_1693"
DATASET_ID="davidsvaughn/math_pairs_1426"


#-------------------------------------------------------------------
# change to the project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
#-------------------------------------------------------------------

# tokenize & save dataset
python siamese_train.py --model_id $MODEL_ID --dataset_id $DATASET_ID --tokenize_only

# Run training
torchrun --nproc_per_node 4 siamese_train.py \
    --model_id $MODEL_ID \
    --dataset_id $DATASET_ID \
    --pooling_mode mean \
    --lm_loss_weight 0.001
