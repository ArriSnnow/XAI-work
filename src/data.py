"""
data.py

Dataset loading utilities for the XAI evaluation project.
Handles Pascal VOC 2007 validation images and ground-truth bounding boxes.
"""

from torchvision.datasets import VOCDetection


def load_voc2007_val(root="./data", download=True):
    """
    Load the Pascal VOC 2007 validation set.

    Parameters
    ----------
    root : str
        Root directory where the VOC dataset is stored or will be downloaded.
    download : bool
        Whether to download the dataset if not present.

    Returns
    -------
    dataset : list
        A list of tuples:
        (PIL.Image, list of bounding boxes),
        where each bounding box is [xmin, ymin, xmax, ymax].
    """
    voc_val = VOCDetection(
        root=root,
        year="2007",
        image_set="val",
        download=download,
        transform=None  # keep raw PIL.Image for custom preprocessing
    )

    dataset = []
    for img_pil, target in voc_val:
        boxes = []

        objs = target["annotation"].get("object", [])
        if isinstance(objs, dict):
            objs = [objs]

        for obj in objs:
            bb = obj["bndbox"]
            boxes.append([
                int(bb["xmin"]),
                int(bb["ymin"]),
                int(bb["xmax"]),
                int(bb["ymax"]),
            ])

        dataset.append((img_pil, boxes))

    return dataset
