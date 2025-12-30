"""
metrics.py

Objective evaluation metrics for attribution maps.
Includes IoU, insertion, deletion, and AUC computation.
"""

import numpy as np
import torch


# ----------------------------
# IoU
# ----------------------------

def compute_iou(binary_map, gt_boxes):
    """
    Compute IoU between a binary attribution map and GT bounding boxes.

    Parameters
    ----------
    binary_map : np.ndarray (H, W)
        Binary attribution map.
    gt_boxes : list
        List of GT boxes [xmin, ymin, xmax, ymax].

    Returns
    -------
    float
        Maximum IoU over all GT boxes.
    """
    h, w = binary_map.shape
    max_iou = 0.0

    for xmin, ymin, xmax, ymax in gt_boxes:
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        gt_mask[ymin:ymax, xmin:xmax] = 1

        intersection = np.logical_and(binary_map, gt_mask).sum()
        union = np.logical_or(binary_map, gt_mask).sum()

        if union > 0:
            iou = intersection / union
            max_iou = max(max_iou, iou)

    return max_iou


def binarize_attribution(attr_map, threshold=0.5):
    """
    Binarize attribution map using a fixed threshold.
    """
    return (attr_map >= threshold).astype(np.uint8)


# ----------------------------
# Deletion / Insertion
# ----------------------------

def deletion_curve(model, input_tensor, attr_map, target_class, steps=50):
    """
    Compute deletion curve.
    """
    model.eval()
    _, _, H, W = input_tensor.shape

    flat_attr = attr_map.flatten()
    sorted_idx = np.argsort(-flat_attr)

    scores = []
    modified = input_tensor.clone()

    for i in range(steps):
        k = int((i + 1) / steps * len(sorted_idx))
        idx = sorted_idx[:k]

        for j in idx:
            y = j // W
            x = j % W
            modified[:, :, y, x] = 0.0

        with torch.no_grad():
            out = model(modified)
            prob = out.softmax(dim=1)[0, target_class].item()
        scores.append(prob)

    return np.array(scores)


def insertion_curve(model, input_tensor, attr_map, target_class, baseline=None, steps=50):
    """
    Compute insertion curve.
    """
    model.eval()
    _, _, H, W = input_tensor.shape

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    flat_attr = attr_map.flatten()
    sorted_idx = np.argsort(-flat_attr)

    scores = []
    modified = baseline.clone()

    for i in range(steps):
        k = int((i + 1) / steps * len(sorted_idx))
        idx = sorted_idx[:k]

        for j in idx:
            y = j // W
            x = j % W
            modified[:, :, y, x] = input_tensor[:, :, y, x]

        with torch.no_grad():
            out = model(modified)
            prob = out.softmax(dim=1)[0, target_class].item()
        scores.append(prob)

    return np.array(scores)


# ----------------------------
# AUC
# ----------------------------

def compute_auc(curve):
    """
    Compute area under curve using trapezoidal rule.
    """
    return np.trapz(curve) / len(curve)

