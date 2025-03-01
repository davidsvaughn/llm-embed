#!/bin/bash

# tokenize & save dataset
python siamese_train.py --dataset_id davidsvaughn/math_pairs_1426

# Run training
torchrun --nproc_per_node 4 siamese_train.py --dataset_id davidsvaughn/math_pairs_1426

# Run training with arguments
# torchrun --nproc_per_node 4 siamese_train.py \
#     --batch_size 4 \
#     --num_epochs 5 \
#     --learning_rate 0.001 \
