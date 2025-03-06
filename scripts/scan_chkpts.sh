#!/bin/bash

# Scan saved checkpoints after finetuning

ITEM_TYPE="bw"
# ITEMS="33234,63166,96340,58566,95508,104462,34002,96326,63172,126288"
# ITEMS="104462,126288"

ITEM_FILTER="n%4==0"

# MODEL_DIR="output6"; CHK_MIN=200; CHK_MAX=700
MODEL_DIR="output8"; CHK_MIN=50; CHK_MAX=4200

POOLING_MODE="lasttoken" # default is "mean"

#-------------------------------------------------------------------
# change to the project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
#-------------------------------------------------------------------

torchrun --nproc_per_node 4 siamese_test.py \
    --item-type $ITEM_TYPE \
    ${POOLING_MODE:+--pooling-mode $POOLING_MODE} \
    scan \
    --model-dir $MODEL_DIR \
    ${ITEMS:+--items $ITEMS} \
    ${ITEM_FILTER:+--item-filter $ITEM_FILTER} \
    ${CHK_LIST:+--chk-list $CHK_LIST} \
    ${CHK_MIN:+--chk-min $CHK_MIN} \
    ${CHK_MAX:+--chk-max $CHK_MAX}
