#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts_conv2d"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times"

# Save to scratch now.
FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_data"

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR
mkdir -p $OUTPUT_DIR

sbatch $SCRIPT_DIR/new_conv2d.sbatch 1