#!/bin/bash
#SBATCH -c 8
#SBATCH -t 7-00:00
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
DATA_DIR="/n/holyscratch01/idreos_lab/Users/azhao/linear_data"
FILE=$DATA_DIR/$1.$2
FINAL_CSV=$HOME_DIR/gpu_profiling/data/linear.$1.$2.csv # Avoid race conditions.

mamba activate $HOME_DIR/env

biases=(0 1)
sizes=(1 2 $(seq 4 4 1024))

# Uncomment for testing purposes
# biases=(1)
# sizes=(1000)

# Create file if it doesn't exist; empties it otherwise.
truncate -s 0 $FINAL_CSV

for bias in "${biases[@]}"
do
    for in_size in "${sizes[@]}"
    do
        for out_size in "${sizes[@]}"
        do

            echo "$1, $2, $bias, $in_size, $out_size-------"
            # Run ncu and export into CSV format for preprocessing.
            ncu --nvtx --nvtx-include "profile_range/" --set full -f --export $FILE --target-processes all $HOME_DIR/env/bin/python3 $SCRIPT_DIR/profile_linear.py $1 $2 $bias $in_size $out_size
            ncu --import $FILE.ncu-rep --csv > $FILE.csv

            # Process the CSV.
            $HOME_DIR/env/bin/python3 $SCRIPT_DIR/parse_ncu.py $FILE.csv $FINAL_CSV $1 $2 $bias $in_size $out_size

        done
    done
done

rm $FILE.ncu-rep
rm $FILE.csv
