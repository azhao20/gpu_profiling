#!/bin/bash

module load python/3.10.12-fasrc01

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR="$HOME_DIR/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/mm_times"
FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/mm

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR
mkdir -p $OUTPUT_DIR

# Runs two sizes; uncomment for testing purposes
# sbatch --array=1-2%30 --export=ALL -o $OUTPUT_DIR/%A_%a.out -e $OUTPUT_DIR/%A_%a.err $SCRIPT_DIR/time_mm.sh

# The full script
sbatch --array=1-76%30 -o $OUTPUT_DIR/%A_%a.out -e $OUTPUT_DIR/%A_%a.err $SCRIPT_DIR/time_mm.sh
