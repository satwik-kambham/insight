#!/bin/bash

rm -f Mambaforge-pypy3-Linux-x86_64.sh

# Install Mambaforge (mamba)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-pypy3-Linux-x86_64.sh"

bash Mambaforge-pypy3-Linux-x86_64.sh -b

~/mambaforge/condabin/conda init

rm -f Mambaforge-pypy3-Linux-x86_64.sh

# Install dependencies
/root/mambaforge/bin/mamba install numpy pandas matplotlib seaborn bokeh -y
mamba install jupyter jupyterlab ipython ipykernel -y

/root/mambaforge/bin/mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia -y

/root/mambaforge/bin/mamba install pytorch-lightning -c conda-forge -y
/root/mambaforge/bin/mamba install torchinfo -c conda-forge -y

/root/mambaforge/bin/mamba install wandb -c conda-forge -y

/root/mambaforge/bin/mamba install -c "nvidia/label/cuda-11.7.0" cuda-nvcc -y

# Update repository
cd insight
git pull
cd ..
