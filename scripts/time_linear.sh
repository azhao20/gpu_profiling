#!/bin/bash
#SBATCH -c 8
#SBATCH -t 1-00:00
#SBATCH -p seas_gpu
#SBATCH --mem=64000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mail-type=END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load python/3.10.12-fasrc01
module load gcc/12.2.0-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load cmake

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts
FINAL_CSV=$HOME_DIR/gpu_profiling/data/linear.time.$1.csv

mamba activate $HOME_DIR/env

sizes=(1 2 $(seq 4 4 124) $(seq 128 8 248) $(seq 256 16 368) $(seq 384 32 480) $(seq 512 64 1024))
precisions=(161 162 32)
biases=(0 1)

# Uncomment for testing purposes
# sizes=(1000)
# precisions=(32)
# biases=(1)

# Create file if it doesn't exist; empties it otherwise.
truncate -s 0 $FINAL_CSV

for precision in "${precisions[@]}"
do
    for bias in "${biases[@]}"
    do
        echo "$precision, $bias--------------" # For some sanity checking.
        for in_size in "${sizes[@]}"
        do
            for out_size in "${sizes[@]}"
            do
                # Process the CSV.
                $HOME_DIR/env/bin/python3 \
                    $SCRIPT_DIR/time_linear.py $1 $precision $bias $in_size $out_size $FINAL_CSV

            done
        done
    done
done
