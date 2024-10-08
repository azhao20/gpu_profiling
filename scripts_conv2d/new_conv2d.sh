#!/bin/bash
#SBATCH -J conv2d
#SBATCH -c 8
#SBATCH --mem=256000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -t 7-00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
SCRIPT_DIR=$HOME_DIR/gpu_profiling/scripts_conv2d

# Save to scratch now.
if [ "$1" = "1" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_backward_data"
elif [ "$1" = "0" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_data"
else
    echo "Invalid input for $1. Expected '0' or '1'."
    exit 1
fi

source $HOME_DIR/gpu_profiling/sh/initconda.sh

# iH, iW, transposed, batch_size
sizes=(2 8 32 128 512 1024)
transposed=(0 1)
batch_sizes=(2 4 8 16 32)

num_iW=${#sizes[@]}
num_transposed=${#transposed[@]}
num_batch_sizes=${#batch_sizes[@]}

# Calculate indices based on SLURM_ARRAY_TASK_ID
index=$(($SLURM_ARRAY_TASK_ID - 1))
iH_index=$(($index / ($num_iW * $num_transposed * $num_batch_sizes)))
temp=$(($index % ($num_iW * $num_transposed * $num_batch_sizes)))
iW_index=$(($temp / ($num_transposed * $num_batch_sizes)))
temp=$(($temp % ($num_transposed * $num_batch_sizes)))
transposed_index=$(($temp / $num_batch_sizes))
batch_size_index=$(($temp % $num_batch_sizes))

# Get the actual values for this job's iteration
iH=${sizes[$iH_index]}
iW=${sizes[$iW_index]}
transposed=${transposed[$transposed_index]}
batch_size=${batch_sizes[$batch_size_index]}

FINAL_CSV=$FINAL_DIR/time.$iH.$iW.$transposed.$batch_size.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

group_sizes=(1 64 128 256 512 1024)
kernel_sizes=(3 5 7)
strides=(1)
dilations=(1)

if [ "$transposed" -eq 1 ]; then
  channel_sizes=(1 2 4 8 16 1024)
  # Transposed convolution case: input_channels == output_channels
  for channels in "${channel_sizes[@]}"; do
    for groups in "${group_sizes[@]}"; do
      for kH in "${kernel_sizes[@]}"; do
        for kW in "${kernel_sizes[@]}"; do
          for stride in "${strides[@]}"; do
            for dilation in "${dilations[@]}"; do
                $HOME_DIR/env/bin/python3 $SCRIPT_DIR/new_conv2d.py \
                    --mode 'time' \
                    --iH $iH \
                    --iW $iW \
                    --b $batch_size \
                    --transposed $transposed \
                    --groups $groups \
                    --kH $kH \
                    --kW $kW \
                    --stride $stride \
                    --dilation $dilation \
                    --backward $1 \
                    --out_file $FINAL_CSV \
                    --in_channels $channels \
                    --out_channels $channels
            done
          done
        done
      done
    done
  done
else
  # Non-transposed convolution case: different input/output channels
  channel_sizes=(1 2 4 8 16 1024)

  for in_channels in "${channel_sizes[@]}"; do
    for out_channels in "${channel_sizes[@]}"; do
      for groups in "${group_sizes[@]}"; do
        for kH in "${kernel_sizes[@]}"; do
          for kW in "${kernel_sizes[@]}"; do
            for stride in "${strides[@]}"; do
              for dilation in "${dilations[@]}"; do
                  $HOME_DIR/env/bin/python3 $SCRIPT_DIR/new_conv2d.py \
                    --mode 'time' \
                    --iH $iH \
                    --iW $iW \
                    --b $batch_size \
                    --transposed $transposed \
                    --groups $groups \
                    --kH $kH \
                    --kW $kW \
                    --stride $stride \
                    --dilation $dilation \
                    --backward $1 \
                    --out_file $FINAL_CSV \
                    --in_channels $in_channels \
                    --out_channels $out_channels
              done
            done
          done
        done
      done
    done
  done
fi
