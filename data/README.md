# Data Description

## Survey
This file contains aggregated statistics corresponding to the human-centered evaluation results reported in the paper.

## Dataset: Pascal VOC 2007 (Validation Set)

This project uses images from the Pascal VOC 2007 validation set for
evaluating explainable AI methods in image classification.

Due to dataset size and licensing considerations, the Pascal VOC dataset
is **not included** in this repository.

The dataset must be downloaded separately from the official source:

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

---

## Dataset Usage

- Split used: **VOC 2007 validation set**
- Images are used to generate visual explanations and to support
  quantitative and human-centered evaluation of explainable AI methods.
- Object bounding boxes are used as reference for localization-based
  evaluation metrics.

---

## Reproducibility

To ensure reproducibility without redistributing the dataset, this
repository provides:

- A list of image IDs used in experiments:
  `voc2007_val_image_ids.txt`
- Jupyter notebooks that assume the standard Pascal VOC directory
  structure under `data/`

Expected directory structure after dataset download:

data/
└── VOCdevkit/
└── VOC2007/
├── JPEGImages/
├── Annotations/
└── ImageSets/
