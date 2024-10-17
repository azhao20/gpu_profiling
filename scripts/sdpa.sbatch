#!/bin/bash

#SBATCH --job-name=sdpa-bench-h100
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=1-00:00
#SBATCH --array=0-55
#SBATCH --mem=64gb
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/job_logs/sdpa/sdpa_%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/job_logs/sdpa/sdpa_%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/

BASE_DATA_DIR=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/data

BACKWARD=(0 1)
DTYPES_B16_16=(b16 16)
BACKENDS_B16_16=(flash efficient cudnn)
NUM_HEADS_B16_16=(4 8 12 16)

DTYPES_FP32=(32)
BACKENDS_FP32=(efficient)
NUM_HEADS_FP32=(4 8 12 16)

# Define the configuration array for b16 and 16 dtypes
configs_b16_16=()
for backward in "${BACKWARD[@]}"; do
    for dtype in "${DTYPES_B16_16[@]}"; do
        for backend in "${BACKENDS_B16_16[@]}"; do
            for num_heads in "${NUM_HEADS_B16_16[@]}"; do
                configs_b16_16+=("$backward $dtype $backend $num_heads")
            done
        done
    done
done

# Define the configuration array for fp32 dtype
configs_fp32=()
for backward in "${BACKWARD[@]}"; do
    for dtype in "${DTYPES_FP32[@]}"; do
        for backend in "${BACKENDS_FP32[@]}"; do
            for num_heads in "${NUM_HEADS_FP32[@]}"; do
                configs_fp32+=("$backward $dtype $backend $num_heads")
            done
        done
    done
done

# Combine the configuration arrays
configs=("${configs_b16_16[@]}" "${configs_fp32[@]}")

# Get the current configuration from the slurm array index
config=(${configs[$SLURM_ARRAY_TASK_ID]})

# Extract the config into named variables
BACKWARD=${config[0]}
DTYPE=${config[1]}
BACKEND=${config[2]}
NUM_HEADS=${config[3]}

# Set the DATA_DIR based on the backward value
if [ $BACKWARD -eq 1 ]; then
    DATA_DIR=$BASE_DATA_DIR/sdpa_backward
else
    DATA_DIR=$BASE_DATA_DIR/sdpa
fi

FINAL_CSV=$DATA_DIR/time.$DTYPE.$BACKEND.$NUM_HEADS.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

# Run the script with the current configuration
srun python scripts/sdpa.py --mode 'time' --dtype $DTYPE --backend $BACKEND --h $NUM_HEADS --backward $BACKWARD --out_file $FINAL_CSV