#!/bin/bash

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/linear_outputs"

module load python/3.10.12-fasrc01

# TODO: find a way to specify different configurations for GPUs.

num_inputs=(1 2 $(seq 4 4 124) $(seq 128 8 248) $(seq 256 16 368) $(seq 384 32 480) $(seq 512 64 1024))

# Uncomment for testing purposes
# num_inputs=(3)

for inputs in "${num_inputs[@]}"
do
    JOB_FILE=$OUTPUT_DIR/${inputs}
            sbatch -o $JOB_FILE.%j.out \
                   -e $JOB_FILE.%j.err \
                   $SCRIPT_DIR/profile_linear.sh $inputs
done

# source $SCRIPT_DIR/run_verify_linear.sh
