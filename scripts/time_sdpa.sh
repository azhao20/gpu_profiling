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

if [ "$4" = "1" ]; then
    FINAL_DIR=$HOME/gpu_profiling/data/final/sdpa_backward
elif [ "$4" = "0" ]; then
    FINAL_DIR=$HOME/gpu_profiling/data/final/sdpa
else
    echo "Invalid input for $4. Expected '0' or '1'."
    exit 1
fi

FINAL_CSV=$FINAL_DIR/time.$1.$2.$3.csv

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

$HOME/env/bin/python3 $SCRIPT_DIR/sdpa.py --mode 'time' --dtype $1 --backend $2 --h $3 --backward $4 --out_file $FINAL_CSV

# <backend>: number of times this script gets invoked * # of params.
# efficient: 4 * (1 + 2) = 12 times
# 12 * 1344 = 16,128
# flash: 4 * 2 = 8 times
# 8 * 9702 = 77,616
# math: 4 * (1 + 2) = 12 times.
# 12 * 9702 = 116,424, if we use the same params for flash
# 12 * 1344 = 16,128, if we use the same params for efficient.
# 16,128 + 77,616 + (116,424 or 16,128) = 210,168 or 109872
# echo "Using backend $1"
# if [ "$1" = "efficient" ]; then # 1344
#     batch_sizes=(2 4 8 16 32 64 128) # 7
#     sq_lengths=(32 64 128 256) # 4
#     skv_lengths=(32 64 128 256) # 4
#     dqk_sizes=(32 64 128) # 3
#     dv_sizes=(32 64 128 256) # 4
# elif [ "$1" = "flash" ]; then # 9702
#     batch_sizes=(2 4 8 16 32 64) # 6
#     sq_lengths=(2 4 8 16 32 64 128 256 512 1024 2048 4096) # 12
#     skv_lengths=(32 64 128 256 512 1024 2048) # 7
#     dqk_sizes=(32 64 128) # 3
#     dv_sizes=(32 64 128 256 512 1024 2048) #7
# else
#     echo "Unsupported backend: $1"
#     exit 1
# fi
# is_causal_flags=(0 1)

# Uncomment for testing.
# batch_sizes=(32)
# sq_lengths=(32)
# skv_lengths=(32)
# dqk_sizes=(32)
# dv_sizes=(32)
# is_causal_flags=(0)

# for b in "${batch_sizes[@]}"
# do
#     for s_q in "${sq_lengths[@]}"
#     do
#         echo "$b, $s_q--------------" # For some sanity checking.
#         for s_kv in "${skv_lengths[@]}"
#         do
#             for d_qk in "${dqk_sizes[@]}"
#             do
#                 for d_v in "${dv_sizes[@]}"
#                 do
#                     $HOME/env/bin/python3 \
#                         $SCRIPT_DIR/sdpa.py --mode 'time' --dtype $1 --backend $2 --b $b --h $2 --s_q $s_q --s_kv $s_kv --d_qk $d_qk --d_v $d_v --out_file $FINAL_CSV
#                 done
#             done
#         done
#     done
# done
