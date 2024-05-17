#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times"

module load python/3.10.12-fasrc01

# TODO: take some of the values from time_conv2d.sh and move them here.
sizes=()

# Uncomment for testing purposes
# sizes=(16)

mkdir -p $OUTPUT_DIR

for n in "${sizes[@]}"
do
    JOB_FILE=$OUTPUT_DIR/$n
    sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_conv2d.sh $n
done
