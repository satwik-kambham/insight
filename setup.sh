#!/bin/bash

# Create and activate environment
mamba create -n insight python=3.10
mamba activate insight

# Install dependencies
mamba install numpy pandas matplotlib seaborn bokeh -y
mamba install jupyter jupyterlab ipython ipykernel -y

mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia -y

mamba install pytorch-lightning -c conda-forge -y
mamba install torchinfo -c conda-forge -y

mamba install wandb -c conda-forge -y

mamba install -c "nvidia/label/cuda-11.7.0" cuda-nvcc -y

# Update repository
cd insight
git pull
cd ..
