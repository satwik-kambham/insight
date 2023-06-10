import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        # inputs: (N, C, H, W)
        # targets: (N, H, W)
        inputs = F.softmax(inputs, dim=1)
        targets = torch.eye(self.num_classes)[targets].permute(0, 3, 1, 2).float()
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice
        return loss.mean()
