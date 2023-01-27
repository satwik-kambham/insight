#!/bin/bash

pip install --upgrade torchinfo wandb torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd insight
git pull
cd ..

wandb login

echo("chmod +x setup.sh")