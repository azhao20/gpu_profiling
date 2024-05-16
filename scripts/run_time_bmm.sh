#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/bmm_times"

module load python/3.10.12-fasrc01

# Up to 512: multiples of 16.
# 512-2048: multiples of 128
# 2048-4096: multiples of 512
# 4096-2^15 = 32768: multiples of 1024
sizes=($(seq 16 16 496) $(seq 512 128 1920) $(seq 2048 512 3584) $(seq 4096 1024 32768))

# Uncomment for testing purposes
sizes=(16)

mkdir -p $OUTPUT_DIR

for n in "${sizes[@]}"
do
    JOB_FILE=$OUTPUT_DIR/$n
    sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_bmm.sh $n
done
