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
FINAL_CSV=$HOME_DIR/gpu_profiling/data/linear.verify.csv

mamba activate $HOME_DIR/env

sizes=(1 2 $(seq 32 32 1024))
precisions=(161 162 32)
biases=(0 1)

# Create file if it doesn't exist; empties it otherwise.
truncate -s 0 $FINAL_CSV

for inputs in "${sizes[@]}"
do
    for precision in "${precisions[@]}"
    do
        for bias in "${biases[@]}"
        do
            echo "$precision, $bias--------------" # For some sanity checking.
            for size in "${sizes[@]}"
            do
                # Process the CSV.
                $HOME_DIR/env/bin/python3 \
                    $SCRIPT_DIR/verify_linear.py $inputs $precision $bias $size $size $FINAL_CSV

            done
        done
    done
done
