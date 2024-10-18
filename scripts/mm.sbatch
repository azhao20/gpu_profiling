#!/bin/bash

#SBATCH --job-name=mm-bench-h100
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --array=2-76
#SBATCH --mem=64gb
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/job_logs/mm/mm_%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/job_logs/mm/mm_%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/

DATA_DIR=/n/holyscratch01/idreos_lab/Users/spurandare/gpu_profiling/experiments/data/mm

# Up to 512: multiples of 16.
# 512-2048: multiples of 128
# 2048-4096: multiples of 512
# 4096-2^15 = 32768: multiples of 1024
sizes=($(seq 16 16 496) $(seq 512 128 1920) $(seq 2048 512 3584) $(seq 4096 1024 32768))
n=${sizes[$SLURM_ARRAY_TASK_ID-1]}
echo $n

FINAL_CSV=$DATA_DIR/time.$n.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

python scripts/mm.py --mode 'time' --n $n --out_file $FINAL_CSV
