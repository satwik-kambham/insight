#!/bin/bash

rm -f Mambaforge-Linux-x86_64.sh

# Install Mambaforge (mamba)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"

bash Mambaforge-Linux-x86_64.sh -b

~/mambaforge/condabin/conda init

rm -f Mambaforge-Linux-x86_64.sh

# Install dependencies
~/mambaforge/bin/mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia -y

~/mambaforge/bin/mamba install networkx -y

~/mambaforge/bin/mamba install pytorch-lightning torchinfo wandb -c conda-forge -y

~/mambaforge/bin/mamba install scipy -c conda-forge -y

~/mambaforge/bin/mamba install -c "nvidia/label/cuda-11.7.0" cuda-nvcc -y

~/mambaforge/bin/mamba install ray-tune optuna -y

# Login to wandb
wandb login

# Update repository
cd insight
git pull
cd ..
