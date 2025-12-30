"""
visualization.py

Visualization utilities for attribution maps.
Handles resizing, colormaps, and overlay rendering.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Color space utilities
# ----------------------------

def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# ----------------------------
# Resizing
# ----------------------------

def resize_attribution(attr_map, image_shape):
    """
    Resize attribution map to match image shape.

    Parameters
    ----------
    attr_map : np.ndarray (H, W)
    image_shape : tuple
        (H, W, C) or (H, W)

    Returns
    -------
    np.ndarray (H, W)
    """
    H, W = image_shape[:2]
    return cv2.resize(attr_map, (W, H))


# ----------------------------
# Heatmap & overlay
# ----------------------------

def apply_colormap(attr_map, colormap=cv2.COLORMAP_JET):
    """
    Apply OpenCV colormap to attribution map.
    """
    heatmap = np.uint8(255 * attr_map)
    return cv2.applyColorMap(heatmap, colormap)


def overlay_heatmap(image_bgr, heatmap_bgr, alpha=0.5):
    """
    Overlay heatmap on image.

    Parameters
    ----------
    image_bgr : np.ndarray (H, W, 3)
    heatmap_bgr : np.ndarray (H, W, 3)
    alpha : float
        Weight for heatmap overlay.

    Returns
    -------
    np.ndarray
    """
    return cv2.addWeighted(heatmap_bgr, alpha, image_bgr, 1 - alpha, 0)


# ----------------------------
# High-level helper
# ----------------------------

def render_attribution(image_bgr, attr_map, alpha=0.5):
    """
    Resize attribution, apply colormap, and overlay on image.

    Returns
    -------
    overlay_bgr : np.ndarray
    """
    attr_resized = resize_attribution(attr_map, image_bgr.shape)
    heatmap = apply_colormap(attr_resized)
    overlay = overlay_heatmap(image_bgr, heatmap, alpha=alpha)
    return overlay


# ----------------------------
# Plotting
# ----------------------------

def show_image(img_rgb, title=None):
    plt.figure()
    plt.imshow(img_rgb)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

