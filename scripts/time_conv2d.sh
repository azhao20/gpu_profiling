#!/bin/bash
#SBATCH -J time_conv2d
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

if [ "$1" = "1" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/conv2d_backward
elif [ "$1" = "0" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/conv2d
else
    echo "Invalid input for $1. Expected '0' or '1'."
    exit 1
fi

source $HOME_DIR/gpu_profiling/sh/initconda.sh

sizes=(2 8 32 128 512 1024) # 1024 for completeness.
transposed=(0 1)

num_iW=${#sizes[@]}
num_transposed=${#transposed[@]}

index=$(($SLURM_ARRAY_TASK_ID - 1))
iH_index=$(($index / ($num_iW * $num_transposed)))
temp=$(($index % ($num_iW * $num_transposed)))
iW_index=$(($temp / $num_transposed))
transposed_index=$(($temp % $num_transposed))

iH=${sizes[$iH_index]}
iW=${sizes[$iW_index]}
transposed=${transposed[$transposed_index]}

FINAL_CSV=$FINAL_DIR/time.$iH.$iW.$transposed.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/conv2d.py --mode 'time' --iH $iH --iW $iW --transposed $transposed --backward $1 --out_file $FINAL_CSV
