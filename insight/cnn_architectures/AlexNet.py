import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()
        self.conv1 = nn.LazyConv2d(96, 11, stride=4)
        self.conv2 = nn.LazyConv2d(256, 5, padding=2)
        self.conv3 = nn.Sequential(
            *[nn.LazyConv2d(i, 3, padding=1) for i in [384, 384, 256]]
        )
        self.pool = nn.MaxPool2d(3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.LazyLinear(4096)
        self.fc3 = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = self.flatten(out)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
