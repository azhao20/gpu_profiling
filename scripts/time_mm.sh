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

HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"
source $HOME/gpu_profiling/sh/initconda.sh

SCRIPT_DIR=$HOME/gpu_profiling/scripts
FINAL_DIR=$HOME/gpu_profiling/data/final/mm
FINAL_CSV=$FINAL_DIR/time.$1.csv

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

$HOME/env/bin/python3 $SCRIPT_DIR/mm.py --mode 'time' --n $1 --out_file $FINAL_CSV

## OLD WORK: MIGHT BE USEFUL FOR PROFILE

# Up to 512: multiples of 16.
# 512-2048: multiples of 128
# 2048-4096: multiples of 512
# 4096-2^15 = 32768: multiples of 1024
# sizes=($(seq 16 16 496) $(seq 512 128 1920) $(seq 2048 512 3584) $(seq 4096 1024 32768))
# dtypes=('b16' '16' '32')

# sizes=(1000)
# dtypes=('32')

# for dtype in "${dtypes[@]}"
# do
#     for m in "${sizes[@]}"
#     do
#         echo "$dtype, $m--------------" # For some sanity checking.
#         for p in "${sizes[@]}"
#         do
#             $HOME/env/bin/python3 \
#                 $SCRIPT_DIR/mm.py --mode 'time' --dtype $dtype --n $1 --m $m --p $p --out_file $FINAL_CSV

#         done
#     done
# done
