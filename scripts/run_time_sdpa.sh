#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/sdpa_times"

module load python/3.10.12-fasrc01

num_heads=(4 8 12 16)

# Uncomment for testing purposes
# num_heads=(8)

mkdir -p $OUTPUT_DIR

dtypes=('b16' '16')
backends=('flash' 'efficient')

# Uncomment for testing purposes
# dtypes=('16')
# backends=('flash')

for dtype in "${dtypes[@]}"
do
    for backend in "${backends[@]}"
    do
        for h in "${num_heads[@]}"
        do
            JOB_FILE=$OUTPUT_DIR/$dtype.$backend.$h
            sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_sdpa.sh $dtype $backend $h $1
        done
    done
done

# fp32 only works with efficient and math.
dtypes=('32')
backends=('efficient')

# Uncomment for testing purposes
# backends=('efficient')

for dtype in "${dtypes[@]}"
do
    for backend in "${backends[@]}"
    do
        for h in "${num_heads[@]}"
        do
            JOB_FILE=$OUTPUT_DIR/$dtype.$backend.$h
            sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_sdpa.sh $dtype $backend $h $1
        done
    done
done