#!/bin/bash

# assumes relevant modules have been loaded, e.g., via initconda.sh
export HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"

DATA_DIR=$HOME/gpu_profiling/data
TRAIN_DIR=$DATA_DIR/new_train_models
# EVAL_DIR=$DATA_DIR/eval_models
FINAL_DIR=$DATA_DIR/final_models

rm -rf $FINAL_DIR
mkdir -p $FINAL_DIR

SCRIPT_DIR=$HOME/gpu_profiling/scripts

# List of data source directories
DATA_SOURCES=("huggingface" "timm" "torchbench")

# List of operation types
OP_TYPES=("addmm" "bmm" "mm" "sdpea" "sdpfa" "conv")

# Iterate over each data source and operation type
for source in "${DATA_SOURCES[@]}"; do
    for op in "${OP_TYPES[@]}"; do
        echo "Processing op_type=$op from source=$source"
        python3 "$SCRIPT_DIR/parse.py" --op_type "$op" --path "$TRAIN_DIR/$source" --save_path "$FINAL_DIR" --overwrite "a"
    done
done