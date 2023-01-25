import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()
        self.conv1 = nn.LazyConv2d(6, 5)
        self.conv2 = nn.LazyConv2d(16, 5)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(num_classes * 4)
        self.fc2 = nn.LazyLinear(num_classes * 2)
        self.fc3 = nn.LazyLinear(num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.avgpool(out)
        out = F.relu(self.conv2(out))
        out = self.avgpool(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
