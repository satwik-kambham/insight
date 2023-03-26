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
    
# Start jupyterlab and expose ports
EXPOSE 8888
CMD "$env_bin/jupyter" lab --allow-root --ip=0.0.0.0 --no-browser --LabApp.trust_xheaders=True --LabApp.disable_check_xsrf=False --LabApp.allow_remote_access=True --LabApp.allow_origin='*'
