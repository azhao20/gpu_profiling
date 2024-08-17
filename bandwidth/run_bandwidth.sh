#!/bin/bash
#SBATCH -J sdpa
#SBATCH -c 8
#SBATCH --mem=256000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SSBATCH -t 0-12:00
#SSBATCH -p gpu_test
#SSBATCH --gres=gpu:1

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/bandwidth
OUT_DIR="$HOME_DIR/gpu_profiling/data/bandwidth"

mkdir -p $OUT_DIR

source $HOME_DIR/gpu_profiling/sh/initconda.sh

$HOME_DIR/env/bin/python3 $SCRIPT_DIR/main.py --gpu $1 --out_dir $OUT_DIR
