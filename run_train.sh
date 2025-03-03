#!/bin/bash

MODEL_ID="microsoft/Phi-4-mini-instruct"
DATASET_ID="davidsvaughn/bw_pairs_1030"

# tokenize & save dataset
python siamese_train.py --model_id $MODEL_ID --dataset_id $DATASET_ID --tokenize_only

# Run training
torchrun --nproc_per_node 4 siamese_train.py \
    --model_id $MODEL_ID \
    --dataset_id $DATASET_ID \
    --pooling_mode mean \
    --lm_loss_weight 0.001
