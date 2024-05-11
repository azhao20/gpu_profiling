module load python/3.10.12-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load gcc/9.5.0-fasrc01

export HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"

export PYTHONPATH=$PYTHONPATH:$HOME/gpu_profiling
export HUGGING_FACE_HUB_TOKEN=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

source activate $HOME/env

# Update
source $HOME/gpu_profiling/sh/.env
# echo $HUGGING_FACE_HUB_TOKEN

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True