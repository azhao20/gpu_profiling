#!/bin/bash
#SBATCH -J run_search
#SBATCH -c 16
#SBATCH --mem=64000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH -p seas_gpu

python jupyter_a100/main.py --op_type $1 --device $2 --n_iter $3
