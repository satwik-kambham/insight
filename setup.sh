#!/bin/bash

rm -f Mambaforge-Linux-x86_64.sh

# Install Mambaforge (mamba)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"

bash Mambaforge-Linux-x86_64.sh -b

~/mambaforge/condabin/conda init

rm -f Mambaforge-Linux-x86_64.sh

# Install dependencies
~/mambaforge/bin/mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
~/mambaforge/bin/mamba install networkx -y
~/mambaforge/bin/mamba install torchmetrics torchinfo wandb -y
~/mambaforge/bin/mamba install scipy -y
~/mambaforge/bin/mamba install -c "nvidia/label/cuda-11.7.0" cuda-nvcc -y
~/mambaforge/bin/mamba install ray-tune optuna -y

~/mambaforge/bin/mamba install ray-tune ipython -y

~/mambaforge/bin/pip install selectivesearch
~/mambaforge/bin/pip install "pytorch-lightning[extra]"

# Update repository
git pull
