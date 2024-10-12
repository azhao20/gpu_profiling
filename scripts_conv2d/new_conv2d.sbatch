#!/bin/bash
#SBATCH -J new_conv2d
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=256000
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times/%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts_conv2d
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 3-00:00
#SBATCH --array=1-720%40

# sizes, transposed, group_sizes, and batch_sizes moved to array indexing
sizes=(32 64 128 224 336 448 512 784 1120) # iH == iW
transposed=(0 1)
group_sizes=(1 16 64 128 256 512 768 1024)
batch_sizes=(2 4 8 16 32)

num_sizes=${#sizes[@]}
num_transposed=${#transposed[@]}
num_groups=${#group_sizes[@]}
num_batches=${#batch_sizes[@]}

# Calculate indices based on SLURM_ARRAY_TASK_ID
index=$(($SLURM_ARRAY_TASK_ID - 1))

size_index=$(($index % $num_sizes))
transposed_index=$((($index / $num_sizes) % $num_transposed))
group_index=$((($index / ($num_sizes * $num_transposed)) % $num_groups))
batch_index=$((($index / ($num_sizes * $num_transposed * $num_groups)) % $num_batches))

iH=${sizes[$size_index]}
iW=$iH  # iH == iW
transposed=${transposed[$transposed_index]}
group_size=${group_sizes[$group_index]}
batch_size=${batch_sizes[$batch_index]}

# Save to scratch.
if [ "$1" = "1" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_backward_data"
elif [ "$1" = "0" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_data"
else
    echo "Invalid input for $1. Expected '0' or '1'."
    exit 1
fi
FINAL_CSV=$FINAL_DIR/time.$iH.$transposed.$group_size.$batch_size.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

# Iterate over the remaining hyperparameters.
kernel_sizes=(3 5 7)
dtypes=("16" "32" "b16")
strides=(1)
dilations=(1)

if [ "$transposed" -eq 1 ]; then
  channel_sizes=(1 8 16 64 128 256 512 768 1024)
  # Transposed convolution case: input_channels == output_channels
  for channels in "${channel_sizes[@]}"; do
    for kH in "${kernel_sizes[@]}"; do
      for stride in "${strides[@]}"; do
        for dilation in "${dilations[@]}"; do
          for dtype in "${dtypes[@]}"; do
            kW=$kH
            python new_conv2d.py \
              --mode 'time' \
              --dtype $dtype \
              --iH $iH \
              --iW $iW \
              --b $batch_size \
              --transposed $transposed \
              --groups $group_size \
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
else
  # Non-transposed convolution case: different input/output channels
  channel_sizes=(1 8 16 64 128 256 512 768 1024)

  for in_channels in "${channel_sizes[@]}"; do
    for out_channels in "${channel_sizes[@]}"; do
      for kH in "${kernel_sizes[@]}"; do
        for stride in "${strides[@]}"; do
          for dilation in "${dilations[@]}"; do
            for dtype in "${dtypes[@]}"; do
              kW=$kH
              python new_conv2d.py \
                --mode 'time' \
                --dtype $dtype \
                --iH $iH \
                --iW $iW \
                --b $batch_size \
                --transposed $transposed \
                --groups $group_size \
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
fi
