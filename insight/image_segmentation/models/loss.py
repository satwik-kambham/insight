import torch
import torch.nn.functional as F


def dice_loss(predicted, target, num_classes, smooth=1e-5):
    # Apply softmax to predicted logits
    predicted = F.softmax(predicted, dim=1)

    # Convert target to one-hot encoding
    target_one_hot = (
        F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    )

    # Flatten predicted and target tensors
    predicted = predicted.reshape(-1, num_classes)
    target_one_hot = target_one_hot.reshape(-1, num_classes)

    # Calculate intersection and union
    intersection = torch.sum(predicted * target_one_hot)
    union = torch.sum(predicted) + torch.sum(target_one_hot)

    # Calculate dice coefficient
    dice_coefficient = (2 * intersection + smooth) / (union + smooth)

    # Calculate dice loss
    dice_loss = 1 - dice_coefficient

    return dice_loss
