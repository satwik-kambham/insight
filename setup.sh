#!/bin/bash

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

# Update repository
cd insight
git pull
cd ..
