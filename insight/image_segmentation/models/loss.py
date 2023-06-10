import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            Shape (N, C, H, W)
        y_true : torch.Tensor
            Shape (N, H, W)
        """
        num_classes = y_pred.shape[1]
        y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
        y_pred = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_true * y_pred, dim=(2, 3))
        cardinality = torch.sum(y_true + y_pred, dim=(2, 3))

        dice_loss = (2.0 * intersection / (cardinality + self.eps)).mean()

        return 1 - dice_loss
