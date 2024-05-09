module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load gcc/9.5.0-fasrc01

conda create -p $(pwd)/env -y python=3.10
source activate $(pwd)/env
conda install -y astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses ccache protobuf numba cython expecttest hypothesis psutil sympy mkl mkl-include git-lfs libpng

conda install -y -c conda-forge tqdm
conda install -y -c huggingface transformers
conda install -y -c pytorch magma-cuda121

python -m pip install triton
python -m pip install pyre-extensions
python -m pip install torchrec
python -m pip install --index-url https://download.pytorch.org/whl/test/ pytorch-triton==3.0.0

export HOME=$(pwd)
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
export DEBUG=0
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
export CC=$(which gcc)
export GCC=$(which gcc)
export GXX=$(which g++)
export CXX=$(which g++)

export CMAKE_GENERATOR="Ninja"
export LD_LIBRARY_PATH=${HOME}/env/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib:${LD_LIBRARY_PATH}
export CUDA_CUDA_LIB="/n/sw/helmod-rocky8/apps/Core/cuda/12.0.1-fasrc01/cuda/lib64/libcudart.so"
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"

# make clone-deps
make pull-deps
make build-deps
