"""
explainers.py

Attribution method implementations for the XAI evaluation project.
Includes Grad-CAM, Saliency, Integrated Gradients, and LIME.
"""

import torch
import numpy as np

# LIME is optional at import time
try:
    from lime import lime_image
except ImportError:
    lime_image = None


# Utility: normalize attribution maps

def normalize_attribution(attr):
    attr = np.array(attr, dtype=np.float32)
    attr = np.maximum(attr, 0)

    min_val = attr.min()
    max_val = attr.max()

    if max_val > min_val:
        attr = (attr - min_val) / (max_val - min_val)
    else:
        attr = np.zeros_like(attr)

    return attr

# Grad-CAM

class GradCAM:
    """
    Grad-CAM implementation.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        return normalize_attribution(cam)


# Saliency Maps

def saliency_map(model, input_tensor, target_class):
    """
    Compute saliency map.
    """
    input_tensor.requires_grad = True
    model.zero_grad()

    output = model(input_tensor)
    score = output[:, target_class]
    score.backward()

    saliency = input_tensor.grad.abs()
    saliency, _ = saliency.max(dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    return normalize_attribution(saliency)


# Integrated Gradients

def integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    """
    Integrated Gradients attribution.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for x in scaled_inputs:
        x.requires_grad = True
        model.zero_grad()
        out = model(x)
        out[:, target_class].backward()
        grads.append(x.grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)
    ig = (input_tensor - baseline) * avg_grads
    ig = ig.abs().max(dim=1)[0].squeeze().cpu().numpy()

    return normalize_attribution(ig)


# LIME
  
def lime_explanation(model, preprocess, image_np, target_class, num_samples=1000):
    """
    LIME image explanation.
    """
    if lime_image is None:
        raise ImportError("lime is not installed")

    def predict(images):
        model.eval()
        imgs = torch.stack([preprocess(img) for img in images])
        with torch.no_grad():
            preds = model(imgs)
        return preds.softmax(dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    mask = explanation.get_image_and_mask(
        label=target_class,
        positive_only=True,
        hide_rest=False,
    )[1]

    return normalize_attribution(mask.astype(np.float32))
