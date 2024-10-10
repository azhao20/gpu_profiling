#!/bin/bash
#SBATCH -c 64
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --mem=256000
#SBATCH --gres=gpu:4
#SBATCH -o torch_install.%j.out
#SBATCH -e torch_install.%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load Mambaforge/23.11.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
module load gcc/12.2.0-fasrc01
module load cmake/3.30.3-fasrc01

function mkdir_if_not_exists {
  if [[ ! -e "$1" ]]; then
      mkdir -p "$1"
  elif [[ ! -d "$1" ]]; then
      echo "$1 already exists but is not a directory, aborting..." 1>&2
      exit 1
  fi
}

function symlink_if_not_exists {
  if [[ ! -e "$2" ]]; then
      ln -s "$1" "$2"
  elif [[ ! -h "$2" ]]; then
      echo "$2 already exists but is not a symbolic link" 1>&2
  fi
}

# Assuming you are at /n/holylabs/LABS/idreos_lab/Users/azhao
HOME_DIR=/n/holylabs/LABS/idreos_lab/Users/azhao
PYTHON_VERSION=3.11

# conda create -y --prefix=pytorch-$PYTHON_VERSION python=$PYTHON_VERSION
eval "$(conda shell.bash hook)"
# conda activate $HOME_DIR/pytorch-$PYTHON_VERSION
# conda install -y astunparse numpy scipy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses protobuf numba cython scikit-learn expecttest hypothesis psutil sympy
# conda install -y -c pytorch magma-cuda124
# conda install -y git-lfs
# conda install -y -c huggingface transformers
# conda install -y -c conda-forge ccache gpustat pwlf librosa tabulate tqdm
# pip uninstall pytorch-triton
# pip install --index-url https://download.pytorch.org/whl/test/ pytorch-triton==3.0.0
# pip install lintrunner

conda deactivate
conda activate $HOME_DIR/pytorch-$PYTHON_VERSION

CCACHE_HOME=$HOME_DIR/ccache
CCACHE_PATH=$CONDA_PREFIX/bin/ccache

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export MAKEFLAGS=-j256
export USE_CUDA=1
export BUILD_CAFFE2_OPS=0
export USE_DISTRIBUTED=1
export MAX_JOBS=40
export USE_CUDNN=1
export USE_NCCL=1
export REL_WITH_DEB_INFO=1
export BUILD_CAFFE2=0
export USE_XNNPACK=0
export USE_MKLDNN=0
export USE_FBGEMM=0
export USE_QNNPACK=0
export USE_NNPACK=0
export BUILD_TEST=0
export USE_GOLD_LINKER=1
export USE_PYTORCH_QNNPACK=0
export DEBUG=0
export BLAS=MKL
export TORCH_CUDA_ARCH_LIST="8.0 9.0"

mkdir_if_not_exists "$CCACHE_HOME"
# Add symlinks
mkdir_if_not_exists "$CCACHE_HOME/bin"
mkdir_if_not_exists "$CCACHE_HOME/lib"
mkdir_if_not_exists "$CCACHE_HOME/cuda"
symlink_if_not_exists "$CCACHE_PATH" "$CCACHE_HOME/bin/ccache"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/cc"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/c++"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/gcc"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/lib/g++"
symlink_if_not_exists "$CCACHE_HOME/bin/ccache" "$CCACHE_HOME/cuda/nvcc"

# Write PATH and CUDA_NVCC_EXECUTABLE to ~/.bashrc so all future shells will have
# ccache set up correctly.
echo "export PATH=$CCACHE_HOME/lib:$CCACHE_HOME/bin:\$PATH" >> ~/.bashrc
echo "export CUDA_NVCC_EXECUTABLE=$CCACHE_HOME/cuda/nvcc" >> ~/.bashrc

# Increase cache size regardless, and fallback to sudo if ccache was installed with
# with sudo permission
"$CCACHE_PATH" -M 25Gi || sudo "$CCACHE_PATH" -M 25Gi

# # Safely derive the exisiting cuda compiler executable's path into variable CUDA_NVCC_EXECUTABLE
CUDA_NVCC_EXECUTABLE=$( command -v nvcc ) || true

# # Add cuda to PATH
if [ ! -x "$CUDA_NVCC_EXECUTABLE" ] || ! (echo "$PATH" | grep -q -F "cuda"); then
  echo "export PATH=$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
  echo "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi

# alias BUILD_VARS='MAKEFLAGS=-j64 USE_LLVM=/usr/lib/llvm-10 USE_CUDA=1 BUILD_CAFFE2_OPS=0 USE_DISTRIBUTED=1 MAX_JOBS=64 USE_CUDNN=1 USE_NCCL=1 REL_WITH_DEB_INFO=1 BUILD_CAFFE2=0 USE_XNNPACK=0 USE_MKLDNN=0 USE_FBGEMM=0 USE_QNNPACK=0 USE_NNPACK=0 BUILD_TEST=0 USE_GOLD_LINKER=1 USE_PYTORCH_QNNPACK=0 DEBUG=0 BLAS=MKL'

make pull-deps
make build-deps
