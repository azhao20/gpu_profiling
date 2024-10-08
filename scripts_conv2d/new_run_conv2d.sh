#!/bin/bash

module load python/3.10.12-fasrc01

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR="$HOME_DIR/gpu_profiling/scripts_conv2d"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times"

# Save to scratch now.
if [ "$1" = "1" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_backward_data"
elif [ "$1" = "0" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_data"
else
    echo "Invalid input for $1. Expected '0' or '1'."
    exit 1
fi

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR
mkdir -p $OUTPUT_DIR

# Runs two sizes; uncomment for testing purposes
# sbatch --array=1-3%2 -o $OUTPUT_DIR/%A_%a.out -e $OUTPUT_DIR/%A_%a.err $SCRIPT_DIR/new_conv2d.sh $1

# The full script
# 1-360%30 => 360 jobs; at most 30 running concurrently.
sbatch --array=1-360%30 -o $OUTPUT_DIR/%A_%a.out -e $OUTPUT_DIR/%A_%a.err $SCRIPT_DIR/new_conv2d.sh $1
