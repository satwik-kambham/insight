import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1 = nn.LazyConv2d(c1, 1)
        self.b2 = nn.Sequential(
            nn.LazyConv2d(c2[0], 1), nn.LazyConv2d(c2[1], 3, padding=1)
        )
        self.b3 = nn.Sequential(
            nn.LazyConv2d(c3[0], 1), nn.LazyConv2d(c3[1], 5, padding=2)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), nn.LazyConv2d(c4, 1)
        )

    def forward(self, x):
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = self.b4(x)
        return torch.cat((out1, out2, out3, out4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.LazyConv2d(64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            nn.LazyConv2d(64, 1),
            nn.LazyConv2d(192, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b3 = nn.Sequential(
            InceptionBlock(64, (96, 128), (16, 32), 32),
            InceptionBlock(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b4 = nn.Sequential(
            InceptionBlock(192, (96, 208), (16, 48), 64),
            InceptionBlock(160, (112, 224), (24, 64), 64),
            InceptionBlock(128, (128, 256), (24, 64), 64),
            InceptionBlock(112, (144, 288), (32, 64), 64),
            InceptionBlock(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.b5 = nn.Sequential(
            InceptionBlock(256, (160, 320), (32, 128), 128),
            InceptionBlock(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def _init_weights(module):
        # Initlize weights with glorot uniform
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
