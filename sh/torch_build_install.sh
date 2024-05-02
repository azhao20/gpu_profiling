#!/bin/bash
#SBATCH -c 64
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --mem=256000
#SBATCH --gres=gpu:4
#SBATCH -o torch_build_install.%j.out
#SBATCH -e torch_build_install.%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load gcc/9.5.0-fasrc01

export HOME=/n/holylabs/LABS/idreos_lab/Users/azhao

mamba remove -p $HOME/env --all
mamba create -p $HOME/env -y python=3.10

mamba deactivate
mamba activate $HOME/env

mamba install -y astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses ccache protobuf numba cython expecttest hypothesis psutil sympy mkl mkl-include git-lfs libpng

mamba install -y -c conda-forge tqdm
mamba install -y -c huggingface transformers
mamba install -y -c pytorch magma-cuda121

python -m pip install triton
python -m pip install pyre-extensions
python -m pip install torchrec
python -m pip install --index-url https://download.pytorch.org/whl/test/ pytorch-triton==3.0.0

export CCACHE_DIR=${HOME}
export CMAKE_PREFIX_PATH=${CONDA_PREFIX}
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
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"

# make pull-deps

echo "Max jobs:"
echo $MAX_JOBS

make build-deps
