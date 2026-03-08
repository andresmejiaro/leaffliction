# Leaffliction

End-to-end plant disease detection pipeline using computer vision and deep learning.

![Leaffliction showcase](assets/showcase.gif)

---

## Overview

Leaffliction classifies leaf images across **8 disease categories** spanning Apple and Grape plants. The pipeline covers dataset analysis, data augmentation, PlantCV-based feature extraction, model training, and inference — all exposed as installable CLI commands.

| Plant  | Classes |
|--------|---------|
| Apple  | Black Rot, Healthy, Rust, Scab |
| Grape  | Black Rot, Esca, Healthy, Spot |

---

## Requirements

- Python 3.10+
- PyTorch (GPU strongly recommended for training)
- CUDA 12.x (optional, for GPU acceleration)

---

## Installation

```bash
pip install -e .
```

This installs the package and registers all CLI commands into your environment.

---

## Pipeline

### 1 — Distribution

Scan a dataset directory and visualize class balance as bar and pie charts.

```bash
Distribution <dataset_dir>
```

Outputs an interactive Plotly chart showing image count per class, useful for spotting imbalances before training.

---

### 2 — Augmentation

Apply 6 geometric transforms to a single image to expand the dataset.

```bash
Augmentation <image_path> [-o <output_dir>]
```

| Transform   | Description |
|-------------|-------------|
| flip        | Horizontal mirror |
| rotate      | 45° counterclockwise |
| crop        | Center quarter crop |
| shear       | Horizontal shear (m = -0.5) |
| distortion  | Affine warp |
| brightness  | +30% brightness |

The augmented images are saved alongside the original and displayed in an interactive grid.

---

### 3 — Transformation

Apply the PlantCV feature extraction pipeline to a single image or a full directory.

```bash
# Single image (opens interactive visualization)
Transformation <image_path>

# Batch processing
Transformation -src <source_dir> -dst <output_dir>
```

Produces 6 outputs per image:

| Stage              | What it shows |
|--------------------|---------------|
| Gaussian Blur      | Noise reduction |
| Mask               | HSV-based green segmentation |
| ROI Objects        | Contour detection overlay |
| Analyze Object     | PlantCV size analysis |
| Pseudo-landmarks   | Skeleton, L/R/T/B/C points, branch tips |
| Masked Color       | Original pixels within leaf mask |

---

### 4 — Train

Run the full training pipeline: augmentation, CSV splitting, model training, checkpoint saving.

```bash
train <dataset_dir> [--epochs 10] [--lr 0.001]
```

Steps performed internally:
1. Collect and augment images to balance classes (5 transforms per image)
2. Split into `train` / `eval` / `test` sets
3. Build `dataset.csv` index
4. Train an `ImageMLP` on 256×256 RGB images
5. Save best checkpoint to `checkpoints/`
6. Export `trained_model.zip`

The model is a fully-connected MLP operating on flattened image tensors with dropout regularization. Training runs on GPU automatically if available.

---

### 5 — Predict

Run inference on a single image and display the prediction with confidence.

```bash
predict <image_path> [-m <checkpoint>] [--device cpu|cuda] [--no-show]
```

Opens a side-by-side Plotly view of the original and preprocessed image with the predicted class and confidence score. `--no-show` skips the window and prints to stdout only.

---

### 6 — Viewer

Sample a random image from the eval split and show the model's prediction vs ground truth.

```bash
viewer [-m <checkpoint>] [--csv dataset.csv] [--split eval]
```

---

### 7 — Metrics

Plot training loss, eval loss, accuracy, sensitivity, and specificity from a saved metrics CSV.

```bash
metrics [training_metrics.csv]
```

---

### 8 — Accuracy

Evaluate the model across an entire directory organized by class folders and print a per-class accuracy table.

```bash
accuracy <dataset_dir> [-m <checkpoint>] [--batch-size 32]
```

Example output:

```
Accuracy: 93.03%
╒══════════════════╤═══════════╤═══════╤══════════╕
│ Class            │   Correct │ Total │ Accuracy │
╞══════════════════╪═══════════╪═══════╪══════════╡
│ Apple_Black_rot  │       ... │   ... │   ...%   │
│ Apple_healthy    │       ... │   ... │   ...%   │
│ ...              │       ... │   ... │   ...%   │
╘══════════════════╧═══════════╧═══════╧══════════╛
```

---

## Project Structure

```
leaffliction/
├── src/leaffliction/
│   ├── cli/                  # CLI entry points
│   │   ├── Augmentation.py
│   │   ├── Distribution.py
│   │   ├── Transformation.py
│   │   ├── accuracy.py
│   │   ├── metrics.py
│   │   ├── predict.py
│   │   ├── train.py
│   │   └── viewer.py
│   ├── Augmentation.py       # Augmentation transforms
│   ├── Transformation.py     # PlantCV feature pipeline
│   ├── accuracy.py           # Batch accuracy evaluation
│   ├── distribution.py       # Dataset statistics
│   ├── metrics.py            # Metrics plotting
│   ├── predict.py            # Inference logic
│   ├── train.py              # Model + training loop
│   └── viewer.py             # Random sample visualizer
├── images/                   # Raw dataset (organized by class)
├── checkpoints/              # Saved model checkpoints
├── assets/
│   └── showcase.gif
├── pyproject.toml
└── requiriments.txt
```

---

## Training Results

| Epoch | Train Loss | Eval Loss | Accuracy | Sensitivity | Specificity |
|-------|-----------|-----------|----------|-------------|-------------|
| 1     | 0.474     | 0.298     | 89.23%   | 89.23%      | 98.46%      |
| 2     | 0.146     | 0.235     | 91.54%   | 91.54%      | 98.79%      |
| 3     | 0.113     | 0.222     | **93.03%** | **93.03%** | **99.00%** |

---

## Authors

Built as a 42 school project.

| | |
|---|---|
| **amejia** | [@andresmejiaro](https://github.com/andresmejiaro) |
| **samusanc** | [@Tagamydev](https://github.com/Tagamydev) |
