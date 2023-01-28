#!/bin/bash

# chmod +x setup.sh

pip install --upgrade torchinfo wandb --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu116

cd insight
git pull
cd ..

wandb login