#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/sdpa_times"

module load python/3.10.12-fasrc01

dims=(32 64 128 256 512 1024 2048)
num_heads=(4 8 12 16)

# Uncomment for testing purposes
dims=(64)
num_heads=(8)

mkdir -p $OUTPUT_DIR

for d_kv in "${dims[@]}"
do
    for h in "${num_heads[@]}"
    do
        JOB_FILE=$OUTPUT_DIR/$d_kv.$h
        sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_sdpa.sh $d_v $h
    done
done

# back * dtype * b * h * d_v * d_qk * s_q * s_kv
# 6 * 4 * 12 * 