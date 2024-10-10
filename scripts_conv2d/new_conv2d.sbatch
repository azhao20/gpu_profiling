#!/bin/bash
#SBATCH -J new_conv2d
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=256000
#SBATCH --array=1-242%30
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times/%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/azhao/conv2d_times/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/scripts_conv2d
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 7-00:00
#SSBATCH -p gpu_test
#SSBATCH --gres=gpu:1
#SSBATCH -t 0-01:00

# iH, iW, transposed
sizes=(32 64 112 128 224 336 384 448 512 784 1120)
transposed=(0 1)

num_iW=${#sizes[@]}
num_transposed=${#transposed[@]}

# Calculate indices based on SLURM_ARRAY_TASK_ID
index=$(($SLURM_ARRAY_TASK_ID - 1))
iH_index=$(($index / ($num_iW * $num_transposed)))
temp=$(($index % ($num_iW * $num_transposed)))
iW_index=$(($temp / $num_transposed))
transposed_index=$(($temp % $num_transposed))

# Get the actual values for this job's iteration
iH=${sizes[$iH_index]}
iW=${sizes[$iW_index]}
transposed=${transposed[$transposed_index]}

# Save to scratch.
if [ "$1" = "1" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_backward_data"
elif [ "$1" = "0" ]; then
    FINAL_DIR="/n/holyscratch01/idreos_lab/Users/azhao/conv2d_data"
else
    echo "Invalid input for $1. Expected '0' or '1'."
    exit 1
fi
FINAL_CSV=$FINAL_DIR/time.$iH.$iW.$transposed.csv
# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

# Iterate over the rest of the hyperparameters.
group_sizes=(1 64 128 256 512 640 768 896 1024)
kernel_sizes=(3 5 7)
dtypes=("16" "32" "b16")
strides=(1)
dilations=(1)
batch_sizes=(2 4 8 16 32)

# Uncomment for testing purposes.
# group_sizes=(1)
# kernel_sizes=(3)
# dtypes=("16")
# batch_sizes=(2)

if [ "$transposed" -eq 1 ]; then
  channel_sizes=(1 8 16 64 128 256 512 640 768 1024)
  # Transposed convolution case: input_channels == output_channels
  for batch_size in "${batch_sizes[@]}"; do
    for channels in "${channel_sizes[@]}"; do
      for groups in "${group_sizes[@]}"; do
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
  done
else
  # Non-transposed convolution case: different input/output channels
  channel_sizes=(1 8 16 64 128 256 512 640 768 1024)

  for batch_size in "${batch_sizes[@]}"; do
    for in_channels in "${channel_sizes[@]}"; do
      for out_channels in "${channel_sizes[@]}"; do
        for groups in "${group_sizes[@]}"; do
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
  done
fi
