#!/bin/bash

SCRIPT_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/sdpa_times"

module load python/3.10.12-fasrc01

num_heads=(4 8 12 16)

# Uncomment for testing purposes
num_heads=(8)

mkdir -p $OUTPUT_DIR

dtypes=('16b' '16')
# TODO: math is C++...?
backends=('flash' 'efficient' 'math')

# Uncomment for testing purposes
dtypes=('16')
backends=('flash')

for dtype in "${half_dtypes[@]}"
do
    for backend in "${backends[@]}"
    do
        for h in "${num_heads[@]}"
        do
            JOB_FILE=$OUTPUT_DIR/$dtype.$backend.$h
            sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_sdpa.sh $dtype $backend $h
        done
    done
done

# fp32 only works with efficient and math.
backends=('efficient' 'math')

# Uncomment for testing purposes
backends=('efficient')

for backend in "${backends[@]}"
do
    for h in "${num_heads[@]}"
    do
        JOB_FILE=$OUTPUT_DIR/32.$backend.$h
        sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/time_sdpa.sh '32' $backend $h
    done
done