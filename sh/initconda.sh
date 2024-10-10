module load Mambaforge/23.11.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
module load gcc/12.2.0-fasrc01

export HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"
export PYTHONPATH="$PYTHONPATH:$HOME/gpu_profiling"

PYTHON_VERSION=3.11
conda deactivate
conda activate $HOME/pytorch-$PYTHON_VERSION

source $HOME/gpu_profiling/sh/.env

# export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/9.1.1.17_cuda12-fasrc01/lib:${LD_LIBRARY_PATH}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
