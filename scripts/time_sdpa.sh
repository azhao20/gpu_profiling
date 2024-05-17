#!/bin/bash
#SBATCH -c 8
#SBATCH -t 7-00:00
#SBATCH -p seas_gpu
#SBATCH --mem=256000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SSBATCH -t 0-12:00
#SSBATCH -p gpu_test
#SSBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu

HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"
source $HOME/gpu_profiling/sh/initconda.sh

SCRIPT_DIR=$HOME/gpu_profiling/scripts
FINAL_DIR=$HOME/gpu_profiling/data/final/sdpa
FINAL_CSV=$FINAL_DIR/time.$1.$2.csv

dtypes=('16b' '16' '32')
# TODO: math is C++...?
backends=('flash' 'efficient' 'math')
batch_sizes=(4 8 16 32 64 128)
# Powers of two - 1.
seq_lengths=(2 4 8 16 32 64 128 256 512 1024 2048)
qk_dims=(32 48 64 128)

# Uncomment for testing.
dtypes=('16')
backends=('efficient')
batch_sizes=(32)
seq_lengths=(32)
qk_dims=(48)

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR

# WARNING: this will delete the CSV if it already exists.
rm $FINAL_CSV

for dtype in "${dtypes[@]}"
do
    for backend in "${backends[@]}"
    do
        echo "$dtype, $backend--------------" # For some sanity checking.
        for b in "${batch_sizes[@]}"
        do
            for s_q in "${seq_lengths[@]}"
            do
                for s_kv in "${seq_lengths[@]}"
                do
                    for d_qk in "${qk_dims[@]}"
                    do
                        $HOME/env/bin/python3 \
                            $SCRIPT_DIR/sdpa.py --mode 'time' --dtype $dtype --b $b --h $2 --s_q $s_q --s_kv $s_kv --d_qk $d_qk --d_v $1 --out_file $FINAL_CSV
                    done
                done
            done
        done
    done
done
