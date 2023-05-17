import numpy as np

import matplotlib
from PIL import Image


def generate_mask_from_class_probabilities(pred, num_classes):
    pred_labels = pred.argmax(dim=1, keepdim=False).numpy()

    # Create color map given number of classes
    color_map = matplotlib.colormaps.get_cmap("hsv")
    color_map = matplotlib.colors.ListedColormap(
        color_map(np.linspace(0, 1, num_classes))
    )

    rgb_mask = color_map(pred_labels[0])
    rgb_mask = (rgb_mask * 255).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_mask)

    return rgb_image
