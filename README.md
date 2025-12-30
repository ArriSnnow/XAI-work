# Human-Centered Evaluation of Visual Explanations

This repository contains the **code and analysis pipeline** for a human-centered evaluation of visual explanation methods (XAI) applied to image classification models.

The project investigates how different explanation techniques are perceived by human users, and how these perceptions relate to commonly used quantitative evaluation metrics.

The repository is intended as a **computational companion** to the associated research paper and focuses on **reproducibility, clarity, and separation of concerns**.

---

## Project Overview

The study compares four widely used visual explanation methods:

- **Grad-CAM**
- **Integrated Gradients**
- **Saliency Maps**
- **LIME**

Explanations are generated for a pretrained image classification model and evaluated through:

1. **Objective metrics** (IoU, insertion, deletion)
2. **Human-centered evaluation** (Likert ratings, preferences, cognitive load)

Only **aggregated results** are included to preserve participant privacy.

---

## Repository Structure

├── src/ # Core implementation (reusable modules)
│ ├── models.py # Model loading and preprocessing
│ ├── data.py # Dataset loading (Pascal VOC 2007)
│ ├── explainers.py # XAI methods (Grad-CAM, IG, Saliency, LIME)
│ ├── metrics.py # Objective evaluation metrics
│ └── visualization.py # Visualization and overlay utilities
│
├── notebooks/ # Reproducible analysis notebooks
│ ├── 01_explanation_generation.ipynb
│ ├── 02_quantitative_metrics.ipynb
│ ├── 03_survey_analysis.ipynb
│ └── 04_figure_generation.ipynb
│
├── data/
│ ├── survey_results_summary.csv # Aggregated survey statistics
│ ├── voc2007_val_image_ids.txt # Image IDs used in experiments
│ └── README.md
│
├── survey/
│ └── consent_text.md # Participant consent and data handling
│
├── README.md
├── LICENSE
└── .gitignore

---

## Notebooks Overview

Each notebook has **one clearly defined responsibility**:

### 1️⃣ `01_explanation_generation.ipynb`
- Loads the pretrained model
- Generates visual explanations for selected images
- Produces normalized attribution maps and visual overlays

### 2️⃣ `02_quantitative_metrics.ipynb`
- Computes objective evaluation metrics:
  - Intersection over Union (IoU)
  - Insertion curves
  - Deletion curves
- Aggregates results across images

### 3️⃣ `03_survey_analysis.ipynb`
- Processes raw survey responses (local only)
- Computes aggregated statistics (means, percentages, rankings)
- Exports a privacy-preserving summary CSV

### 4️⃣ `04_figure_generation.ipynb`
- Reproduces figures used in the paper
- Uses only precomputed results and aggregated data
- Contains no experimental logic

---

## Dataset

- **Images:** Pascal VOC 2007 validation set  
- Images are downloaded automatically via `torchvision` when required.
- A fixed subset of image IDs used in the experiments is provided for reproducibility.

No image data is stored in this repository.

---

## Survey Data and Ethics

- Participation was voluntary and anonymous.
- No personally identifiable information was collected.
- Raw survey responses are **not publicly released**.
- Only aggregated statistics are included (`data/survey_results_summary.csv`).

See `survey/consent_text.md` for details.

---

## Reproducibility

To reproduce the experiments:

1. Clone the repository
2. Install dependencies (PyTorch, torchvision, numpy, pandas, matplotlib, lime)
3. Run notebooks in numerical order (`01` → `04`)

Each notebook is self-contained and relies on the shared `src/` modules.

---

## Notes

- This repository intentionally **does not include** the manuscript or raw survey instruments.
- The focus is on the **computational pipeline** and analysis.
- The code is structured to support future extensions (e.g., additional XAI methods or evaluation protocols).

---

## License

This project is released under the license specified in the `LICENSE` file.
