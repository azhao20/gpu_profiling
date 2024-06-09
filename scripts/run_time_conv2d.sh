#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times"

module load python/3.10.12-fasrc01

# Spin up 2 * 6^2 = 72 jobs.
sizes=(2 8 32 128 512 1024) # 1024 for completeness.
transposed=(0 1)

# Uncomment for testing purposes
sizes=(64)
transposed=(0)

mkdir -p $OUTPUT_DIR

for iH in "${sizes[@]}"
do
    for iW in "${sizes[@]}"
    do
        for transposed in "${transposed[@]}"
        do
            JOB_FILE=$OUTPUT_DIR/$iH.$iW
            sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_conv2d.sh $iH $iW $transposed $1
        done
    done
done
