#!/bin/bash

# chmod +x setup.sh

pip install --upgrade --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu116

pip install torchinfo wandb

cd insight
git pull
cd ..

wandb login