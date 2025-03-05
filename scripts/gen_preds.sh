#!/bin/bash

# Generate and save score predictions prior to building pairwise dataset
# - predictions are used to determine "hard" pairs for training...
# - "hard negatives" ~= false positives (true scores different, predicted scores similar)
# - "hard positives" ~= false negatives (true scores similar, predicted scores different)

ITEM_TYPE="bw"

MODEL_DIR="~/models"

# MODEL_ID="phi4-bw-1"; POOLING_MODE="lasttoken"
MODEL_ID="dan-bw"; POOLING_MODE="mean"
# MODEL_ID="dan-bw-exl2-q4"; POOLING_MODE="mean" # default is "mean"

ITEM_FILTER="n%2!=0"
# ITEMS="33234,63166"

HH_MIN=0.6

#-------------------------------------------------------------------
# change to the project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
#-------------------------------------------------------------------

torchrun --nproc_per_node 4 siamese_test.py \
    --item-type $ITEM_TYPE \
    ${POOLING_MODE:+--pooling-mode $POOLING_MODE} \
    ${HH_MIN:+--hh-min $HH_MIN} \
    gen \
    --model-dir $MODEL_DIR \
    --model-id $MODEL_ID \
    ${ITEM_FILTER:+--item-filter $ITEM_FILTER} \
    ${ITEMS:+--items $ITEMS} \
