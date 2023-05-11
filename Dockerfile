# Set base image
FROM ubuntu:22.04

# Install required apt packages
RUN apt update
RUN apt install -y --no-install-recommends \
    git \
    ca-certificates \
    wget


# Download and install mambaforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
RUN bash Mambaforge-Linux-x86_64.sh -b
RUN rm -f Mambaforge-Linux-x86_64.sh

# Copy over environment yml specification
WORKDIR /insight
COPY environment.yml .

# Create and activate new environment from yml specification
ENV env_name=insight
ENV env_bin=/root/mambaforge/envs/$env_name/bin
RUN ~/mambaforge/condabin/conda init
RUN ~/mambaforge/condabin/mamba env create --file environment.yml
RUN rm -f environment.yml

# Get latest version of insight from github
WORKDIR /notebooks
RUN git clone https://github.com/code-explorer/insight.git

# Install package using setuptools in editable mode
WORKDIR /notebooks/insight
RUN $env_bin/pip install -e .

# Activate insight environment
RUN echo "conda activate insight" >> ~/.bashrc

# Set bash as default shell
SHELL ["/bin/bash", "-c"]
