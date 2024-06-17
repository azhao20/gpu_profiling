#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=256000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -t 7-00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SSBATCH -t 0-12:00
#SSBATCH -p gpu_test
#SSBATCH --gres=gpu:1

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts

if [ "$4" = "1" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/conv2d_backward
elif [ "$4" = "0" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/conv2d
else
    echo "Invalid input for $4. Expected '0' or '1'."
    exit 1
fi

FINAL_CSV=$FINAL_DIR/time.$1.$2.$3.csv

source $HOME_DIR/gpu_profiling/sh/initconda.sh

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/conv2d.py --mode 'time' --iH $1 --iW $2 --transposed $3 --backward $4 --out_file $FINAL_CSV
