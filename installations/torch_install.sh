#!/bin/bash
#SBATCH -c 8
#SBATCH -t 0-06:00
#SBATCH -p seas_gpu
#SBATCH --mem=256000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH -o torch_install.out
#SBATCH -e torch_install.err
#SBATCH --mail-type=END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load gcc/12.2.0-fasrc01

mamba create -p $(pwd)/env -y python=3.10
mamba activate $(pwd)/env

module load cmake
mamba install -y astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses ccache protobuf numba cython expecttest hypothesis psutil sympy mkl mkl-include git-lfs libpng

mamba install -y -c conda-forge tqdm
mamba install -y -c huggingface transformers
mamba install -y -c pytorch magma-cuda121
