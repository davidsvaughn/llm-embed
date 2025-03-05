#!/bin/bash

# Check if required MODEL_PATH argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 MODEL_PATH [BITS]"
    echo "  MODEL_PATH: Path to the model directory"
    echo "  BITS: Bit precision for quantization (default: 4)"
    exit 1
fi

#-------------------------------------------------------------------
# change to the project root directory (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
#-------------------------------------------------------------------

# Process arguments
MODEL_PATH=$1

# Set default value for BITS if not provided
BITS=${2:-4}

# Expand tilde to home directory if present
if [[ "$MODEL_PATH" == "~"* ]]; then
    MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
fi

# Convert MODEL_PATH to absolute path if it's relative
if [[ ! "$MODEL_PATH" = /* ]]; then
    MODEL_PATH="$(pwd)/$MODEL_PATH"
fi

# Remove trailing slash if present
MODEL_PATH=${MODEL_PATH%/}

# Print the parameters
echo "Using model path: $MODEL_PATH"
echo "Using bits: $BITS"

# create fresh TMP directory
TMP="/tmp/exl2"
if [ -d $TMP ]; then
    sudo rm -rf $TMP
fi
mkdir $TMP
mkdir -p $MODEL_PATH/exl2

# create measurement file
MEAS_FILE="$MODEL_PATH/exl2/measurement.json"
if [ -f $MEAS_FILE ]; then
    echo "Measurement file already exists: $MEAS_FILE"
else
    python exllamav2/convert.py \
        -i $MODEL_PATH/ \
        -o $TMP/ \
        -nr \
        -om $MEAS_FILE
fi

# quantize model
QUANT_MODEL_PATH="$MODEL_PATH-exl2-$BITS"bit
if [ -d $QUANT_MODEL_PATH ]; then
    echo "Quantized model directory already exists: $QUANT_MODEL_PATH"
else
    python exllamav2/convert.py \
        -i $MODEL_PATH/ \
        -o $TMP/ \
        -nr \
        -m $MEAS_FILE \
        -cf $QUANT_MODEL_PATH/ -b $BITS
fi