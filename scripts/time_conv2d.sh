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
FINAL_DIR=$HOME/gpu_profiling/data/final/bmm
FINAL_CSV=$FINAL_DIR/time.$1.csv

# 3*5*11*11^2*8*6*4 * 2 = 7-8 million...
dtypes=('16b' '16' '32')
batch_sizes=(2 4 8 16 32)
in_channel_sizes=(4 8 16 32 64 128 256 512 1024 2048 4096)
sizes=(1 2 4 8 16 32 64 128 256 512 1024)
out_channel_sizes=(16 32 64 128 256 512 1024 2048)
group_sizes=(1 64 128 256 512 1024)
kernel_sizes=(1 3 5 7)
transposed=(0 1)
# dilation_sizes=(1)

dtypes=('32')
batch_sizes=(16)
in_channel_sizes=(2)
sizes=(32)
out_channel_sizes=(32)
group_sizes=(32)
kernel_sizes=(3)
transposed=(0)
# dilation_sizes=(1)

# -p: ok if directory already exists.
mkdir -p $FINAL_DIR

# WARNING: this will delete the CSV if it already exists.
rm $FINAL_CSV

for dtype in "${dtypes[@]}"
do
    for b in "${batch_sizes[@]}"
    do
        for in_channels in "${in_channel_sizes[@]}"
        do
            echo "$dtype, $b, $in_channels--------------" # For some sanity checking.
            for iH in "${sizes[@]}"
            do
                for iW in "${sizes[@]}"
                do
                    for out_channels in "${out_channel_sizes[@]}"
                    do
                        for groups in "${group_sizes[@]}"
                        do
                            for kH in "${kernel_sizes[@]}"
                            do
                                for use_T in "${transposed[@]}"
                                do
                                    $HOME/env/bin/python3 \
                                        $SCRIPT_DIR/conv2d.py --mode 'time' --dtype $dtype --b $b --in_channels $in_channels --iH $iH --iW $iW --out_channels $out_channels --groups 1 --kh $kH --kW $kW --stride 1 --dilation 1 --transposed $use_T --out_file $FINAL_CSV
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
