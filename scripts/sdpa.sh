#!/bin/bash
#SBATCH -J sdpa
#SBATCH -c 8
#SBATCH --mem=256000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -t 3-00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SSBATCH -t 0-12:00
#SSBATCH -p gpu_test
#SSBATCH --gres=gpu:1

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"

if [ "$4" = "1" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/sdpa_backward
elif [ "$4" = "0" ]; then
    FINAL_DIR=$HOME_DIR/gpu_profiling/data/final/sdpa
else
    echo "Invalid input for $4. Expected '0' or '1'."
    exit 1
fi

FINAL_CSV=$FINAL_DIR/time.$1.$2.$3.csv

# WARNING: this will delete the CSV if it already exists.
if [ -f "$FINAL_CSV" ]; then
    echo "Deleting file $FINAL_CSV"
    rm "$FINAL_CSV"
fi

python3 scripts/sdpa.py --mode 'time' --dtype $1 --backend $2 --h $3 --backward $4 --out_file $FINAL_CSV
