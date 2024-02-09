#!/bin/bash

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts
OUTPUT_DIR="/n/holyscratch01/idreos_lab/Users/azhao/linear_outputs"

module load python/3.10.12-fasrc01

# TODO: find a way to specify different configurations for GPUs.

precisions=(161 162 32)
num_inputs=(1 2 $(seq 4 4 1024))

# Uncomment for testing purposes
# precisions=(32)
# num_inputs=(3)

for precision in "${precisions[@]}"
do
    for inputs in "${num_inputs[@]}"
    do
        JOB_FILE=$OUTPUT_DIR/${precision}.${inputs}
        sbatch -o $JOB_FILE.%j.out \
               -e $JOB_FILE.%j.err \
               $SCRIPT_DIR/profile_linear.sh $precision $inputs
    done
done
