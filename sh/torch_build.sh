#!/bin/bash
#SBATCH -c 32
#SBATCH -t 7-00:00
#SBATCH -p seas_gpu
#SBATCH --mem=256000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH -o torch_build.%j.out
#SBATCH -e torch_build.%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load python/3.10.12-fasrc01
module load gcc/9.5.0-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

export HOME=/n/holylabs/LABS/idreos_lab/Users/azhao

mamba deactivate
mamba activate $HOME/env

module load cmake/3.28.3-fasrc01

export CCACHE_DI2=${HOME}/ccache
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which mamba))/../"}
export CUDA_NVCC_EXECUTABLE=${CUDA_HOME}/bin/nvcc
export USE_CUDA=1
export MAX_JOBS=40
export USE_CUDNN=1
export USE_NCCL=1
export REL_WITH_DEB_INFO=1
export BUILD_CAFFE2=0
export USE_XNNPACK=0
export USE_FBGEMM=0
export USE_QNNPACK=0
export USE_NNPACK=0
export BUILD_TEST=0
export USE_GOLD_LINKER=1
export USE_PYTORCH_QNNPACK=0
export USE_KINETO=0 # For NCU
export DEBUG=0
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
export CC=$(which gcc)
export GCC=$(which gcc)
export GXX=$(which g++)
export CXX=$(which g++)

export CMAKE_GENERATOR="Ninja"
export LD_LIBRARY_PATH=${HOME}/env/lib:${LD_LIBRARY_PATH}
export CUDA_CUDA_LIB="/n/sw/helmod-rocky8/apps/Core/cuda/12.0.1-fasrc01/cuda/lib64/libcudart.so"
export TORCH_CUDA_ARCH_LIST="7.0 8.0"

# make clone-deps
make pull-deps
make build-deps