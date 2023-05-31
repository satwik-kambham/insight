import numpy as np

import matplotlib
from PIL import Image


def generate_mask(pred, num_classes, one_hot=True):
    if one_hot:
        pred_labels = pred.argmax(dim=1, keepdim=False).cpu().numpy()
    else:
        pred_labels = pred.cpu().numpy()

    # Create color map given number of classes
    color_map = matplotlib.colormaps.get_cmap("gnuplot2")
    color_map = matplotlib.colors.ListedColormap(
        color_map(np.linspace(0, 1, num_classes))
    )

    rgb_mask = color_map(pred_labels[0] / num_classes)
    rgb_mask = (rgb_mask[:, :, :3] * 255).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_mask)

    return rgb_image
