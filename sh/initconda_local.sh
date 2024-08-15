export HOME="/Users/andrew/Desktop/Harvard/idreos-research"
export PYTHONPATH="$PYTHONPATH:$HOME/gpu_profiling"

conda deactivate
conda activate research

source $HOME/gpu_profiling/sh/.env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
