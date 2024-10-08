#!/bin/bash

module load python/3.10.12-fasrc01

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR="$HOME_DIR/gpu_profiling/scripts"
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/sdpa_times"

if [ "$1" = "1" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/sdpa_backward
elif [ "$1" = "0" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/sdpa
else
    echo "Invalid input for $1. Expected '0' or '1'."
    exit 1
fi

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR
mkdir -p $OUTPUT_DIR

num_heads=(4 8 12 16)
dtypes=('b16' '16')
backends=('flash' 'efficient' 'cudnn')

# Uncomment for testing purposes
# num_heads=(8)
# dtypes=('16')
# backends=('flash')

for dtype in "${dtypes[@]}"
do
    for backend in "${backends[@]}"
    do
        for h in "${num_heads[@]}"
        do
            JOB_FILE=$OUTPUT_DIR/$dtype.$backend.$h
            sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/sdpa.sh $dtype $backend $h $1
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
            sbatch -o $JOB_FILE.%j.out -e $JOB_FILE.%j.err $SCRIPT_DIR/sdpa.sh $dtype $backend $h $1
        done
    done
done