#!/bin/bash

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts
JOB_FILE="/n/holyscratch01/idreos_lab/Users/azhao/linear_outputs/verify"

module load python/3.10.12-fasrc01

# TODO: find a way to specify different configurations for GPUs.

sbatch -o $JOB_FILE.%j.out \
       -e $JOB_FILE.%j.err \
       $SCRIPT_DIR/verify_linear.sh