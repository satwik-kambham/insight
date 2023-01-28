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
        for block in config:
            for i, (out_channels, kernel_size) in enumerate(block):
                blocks[f"conv{i+1}"] = nn.LazyConv2d(
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
                blocks[f"relu{i+1}"] = nn.ReLU()
            blocks["pool"] = nn.MaxPool2d(2, 2)

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
        self.blocks = nn.Sequential(blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out
