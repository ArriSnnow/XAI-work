"""
models.py

Model loading and preprocessing utilities for the XAI evaluation project.
"""

import torch
from torchvision.models import resnet50, ResNet50_Weights


def get_device():
    """Return the computation device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    return device


def load_model(device=None):
    """
    Load the pretrained ResNet-50 model used in the study.
    """
    if device is None:
        device = get_device()

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()
    return model


def get_preprocess():
    """
    Return the preprocessing pipeline associated with the pretrained weights.
    """
    weights = ResNet50_Weights.DEFAULT
    return weights.transforms()


def get_target_layer(model, layer_type="late"):
    """
    Return the target layer used for attribution methods.

    Parameters
    ----------
    layer_type : str
        "early" -> model.layer1[0]
        "late"  -> model.layer4[-1]
    """
    if layer_type == "early":
        return model.layer1[0]
    elif layer_type == "late":
        return model.layer4[-1]
    else:
        raise ValueError("layer_type must be 'early' or 'late'")

