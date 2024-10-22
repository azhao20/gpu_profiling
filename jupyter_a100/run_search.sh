#!/bin/bash
#SBATCH -J run_search
#SBATCH -c 16
#SBATCH --mem=256000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH -p seas_compute
#SBATCH -o /n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/%A_%a.out
#SBATCH -e /n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/%A_%a.err

python jupyter_a100/main.py --op_type $1 --device $2 --n_iter $3
