#!/bin/bash

#SBATCH --job-name=conv2d-bench-h100
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=1-00:00
#SBATCH --array=0-1439%60
#SBATCH --mem=64gb
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/job_logs/conv2d/conv2d_%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/job_logs/conv2d/conv2d_%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/

BASE_DATA_DIR=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/data

sizes=(32 64 128 224 336 448 512 784 1120) # iH == iW
transposed=(0 1)
group_sizes=(1 16 64 128 256 512 768 1024)
batch_sizes=(2 4 8 16 32)
backward=(0 1)

configs=()
for backward in "${backward[@]}"; do
  for size in "${sizes[@]}"; do
    for transposed in "${transposed[@]}"; do
      for group_size in "${group_sizes[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
          configs+=("$backward $size $transposed $group_size $batch_size")
        done
      done
    done
  done
done

# Get the current configuration from the slurm array index
config=(${configs[$SLURM_ARRAY_TASK_ID]})

# Extract the config into named variables
BACKWARD=${config[0]}
IMAGE_SIZE=${config[1]}
TRANSPOSED=${config[2]}
GROUP_SIZE=${config[3]}
BATCH_SIZE=${config[4]}

# Save to scratch.
if [ "$BACKWARD" -eq 1 ]; then
  DATA_DIR=$BASE_DATA_DIR/conv2d_backward
else
  DATA_DIR=$BASE_DATA_DIR/conv2d
fi
FINAL_CSV=$DATA_DIR/time.$IMAGE_SIZE.$TRANSPOSED.$GROUP_SIZE.$BATCH_SIZE.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
  echo "Deleting file $FINAL_CSV"
  rm "$FINAL_CSV"
fi

# Iterate over the remaining hyperparameters.
kernel_sizes=(3 5 7)
dtypes=("16" "32" "b16")
strides=(1 2)
dilations=(1)

if [ "$TRANSPOSED" -eq 1 ]; then
  channel_sizes=(1 8 16 64 128 256 512 768 1024)
  # Transposed convolution case: input_channels == output_channels
  for channels in "${channel_sizes[@]}"; do
    for kH in "${kernel_sizes[@]}"; do
      for stride in "${strides[@]}"; do
        for dilation in "${dilations[@]}"; do
          for dtype in "${dtypes[@]}"; do
            kW=$kH
            srun python scripts_conv2d/new_conv2d.py \
              --mode 'time' \
              --dtype $dtype \
              --iH $IMAGE_SIZE \
              --iW $IMAGE_SIZE \
              --b $BATCH_SIZE \
              --transposed $TRANSPOSED \
              --groups $GROUP_SIZE \
              --kH $kH \
              --kW $kW \
              --stride $stride \
              --dilation $dilation \
              --backward $BACKWARD \
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
              srun python scripts_conv2d/new_conv2d.py \
                --mode 'time' \
                --dtype $dtype \
                --iH $IMAGE_SIZE \
                --iW $IMAGE_SIZE \
                --b $BATCH_SIZE \
                --transposed $TRANSPOSED \
                --groups $GROUP_SIZE \
                --kH $kH \
                --kW $kW \
                --stride $stride \
                --dilation $dilation \
                --backward $BACKWARD \
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