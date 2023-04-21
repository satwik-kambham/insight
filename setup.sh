#!/bin/bash

# Install Mambaforge (mamba)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"

bash Mambaforge-Linux-x86_64.sh -b
~/mambaforge/condabin/conda init
rm -f Mambaforge-Linux-x86_64.sh

# Install dependencies
~/mambaforge/bin/mamba env create --file environment.yml
