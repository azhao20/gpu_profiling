#!/bin/bash
#SBATCH -J time_bmm
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
SCRIPT_DIR="$HOME_DIR/gpu_profiling/scripts"
FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/bmm

source $HOME_DIR/gpu_profiling/sh/initconda.sh

# Up to 512: multiples of 16.
# 512-2048: multiples of 128
# 2048-4096: multiples of 512
# 4096-2^15 = 32768: multiples of 1024
sizes=($(seq 16 16 496) $(seq 512 128 1920) $(seq 2048 512 3584) $(seq 4096 1024 32768))
n=${sizes[$SLURM_ARRAY_TASK_ID-1]}
echo $n

FINAL_CSV=$FINAL_DIR/time.$n.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/bmm.py --mode 'time' --n $n --out_file $FINAL_CSV
