#!/bin/bash

DATASET_ID="davidsvaughn/math_pairs_1426"
MODEL_ID="microsoft/Phi-4-mini-instruct"

# tokenize & save dataset
python siamese_train.py --dataset_id $DATASET_ID --model_id $MODEL_ID --tokenize_only

# Run training
torchrun --nproc_per_node 4 siamese_train.py \
    --dataset_id $DATASET_ID \
    --model_id $MODEL_ID \
    --pooling_mode lasttoken \
    --lm_loss_weight 0.01
