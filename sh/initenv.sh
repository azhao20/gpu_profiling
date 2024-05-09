#!/bin/bash

# Old order.
# module load cuda/12.0.1-fasrc01
# module load cudnn/8.9.2.26_cuda12-fasrc01
# module load gcc/9.5.0-fasrc01
# module load cmake

# New order.
module load python/3.10.12-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load gcc/9.5.0-fasrc01
# module load cmake

HOME_DIR="/n/holylabs/LABS/idreos_lab/Users/azhao"
DATA_DIR="/n/holyscratch01/idreos_lab/Users/azhao"

mamba deactivate
mamba activate $HOME_DIR/env

# export TORCHINDUCTOR_PROFILE=1

# ncu --set full --export $HOME_DIR/ncu_data/profile_results --target-processes all $(which python3) profile_test.py

# ncu --target-processes all $(which python3) profile_test.py
