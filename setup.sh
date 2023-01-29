#!/bin/bash

rm -f Mambaforge-pypy3-Linux-x86_64.sh

# Install Mambaforge (mamba)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-pypy3-Linux-x86_64.sh"

bash Mambaforge-pypy3-Linux-x86_64.sh

# Create and activate environment
mamba create -n insight python=3.10
mamba activate insight

# Install dependencies
mamba install numpy pandas matplotlib seaborn bokeh
mamba install jupyter jupyterlab ipython ipykernel

mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia

mamba install pytorch-lightning -c conda-forge
mamba install torchinfo -c conda-forge

mamba install wandb -c conda-forge

conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc
