from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


VGG_A = (
    ((64, 3),),
    ((128, 3),),
    ((256, 3), (256, 3)),
    ((512, 3), (512, 3)),
    ((512, 3), (512, 3)),
)

VGG_B = (
    ((64, 3), (64, 3)),
    ((128, 3), (128, 3)),
    ((256, 3), (256, 3)),
    ((512, 3), (512, 3)),
    ((512, 3), (512, 3)),
)

VGG_C = (
    ((64, 3), (64, 3)),
    ((128, 3), (128, 3)),
    ((256, 3), (256, 3), (256, 1)),
    ((512, 3), (512, 3), (512, 1)),
    ((512, 3), (512, 3), (512, 1)),
)

VGG_D = (
    ((64, 3), (64, 3)),
    ((128, 3), (128, 3)),
    ((256, 3), (256, 3), (256, 3)),
    ((512, 3), (512, 3), (512, 3)),
    ((512, 3), (512, 3), (512, 3)),
)

VGG_E = (
    ((64, 3), (64, 3)),
    ((128, 3), (128, 3)),
    ((256, 3), (256, 3), (256, 3), (256, 3)),
    ((512, 3), (512, 3), (512, 3), (512, 3)),
    ((512, 3), (512, 3), (512, 3), (512, 3)),
)


class VGG(nn.Module):
    def __init__(self, config=VGG_A, num_classes=256):
        super().__init__()
        blocks = {}
        for i, block_config in enumerate(config):
            block = {}
            for j, (out_channels, kernel_size) in enumerate(block_config):
                block[f"conv{j+1}"] = nn.LazyConv2d(
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
                block[f"relu{j+1}"] = nn.ReLU()
            block["pool"] = nn.MaxPool2d(2, 2)
            blocks[f"block{i+1}"] = nn.Sequential(OrderedDict(block))

        blocks["fc"] = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(num_classes),
        )
        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.apply(self._init_weights)

    def forward(self, x):
        out = self.blocks(x)
        return out

    def _init_weights(self):
        # Initlize weights with glorot uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
