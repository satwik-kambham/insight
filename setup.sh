#!/bin/bash

# chmod +x setup.sh

cd insight
git pull
cd ..

pip install --upgrade --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117

pip install torchinfo wandb

wandb login