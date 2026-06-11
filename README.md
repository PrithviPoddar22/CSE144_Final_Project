# CSE 144 Final Project — Transfer Learning Challenge
**UC Santa Cruz | Spring 2026**  
**Author:** Prithvi Poddar

---

## Kaggle Leaderboard

<img width="1509" height="204" alt="image" src="https://github.com/user-attachments/assets/fee48ef3-e3e1-4e0a-a2b6-52b20c151d1e" />


**Score: 0.62727 | Rank: 51 | Team: Prithvi Poddar**

---

## Overview

100-class image classification using transfer learning on a limited dataset (~10 images per class). Fine-tuned a pretrained EfficientNet-B2 backbone with aggressive data augmentation to achieve 62.7% test accuracy, exceeding the 60% baseline.

---

## Repository Structure

```
CSE144-Final-Project/
├── train.py              # Full training and inference pipeline
├── report.pdf            # Project report
├── leaderboard.png       # Kaggle leaderboard screenshot
└── README.md
```

---

## Model Weights

Trained model weights (`best_model.pth`) are hosted on Google Drive:  
**[Download best_model.pth]([https://drive.google.com/file/d/1K6KPXHZtVucujbZKzsnHv9pmMIVgPSff/view?usp=drive_link](https://drive.google.com/file/d/1K6KPXHZtVucujbZKzsnHv9pmMIVgPSff/view?usp=sharing))**

---

## Setup

```bash
pip install torch torchvision pandas pillow
```

---

## Training

```bash
python train.py \
  --mode train \
  --data_dir data/ \
  --output_dir outputs/ \
  --epochs 30 \
  --backbone efficientnet_b2
```

Expected output: model saved to `outputs/best_model.pth`

---

## Inference

```bash
python train.py \
  --mode test \
  --data_dir data/ \
  --weights outputs/best_model.pth \
  --output_dir outputs/
```

Then fix the submission ID format before uploading to Kaggle:

```python
import pandas as pd
df = pd.read_csv('outputs/submission.csv')
df['ID'] = df['ID'].astype(int).astype(str) + '.jpg'
df.to_csv('outputs/submission_fixed.csv', index=False)
```

Submit `outputs/submission_fixed.csv` to Kaggle.

---

## Dataset

Download from the Kaggle competition page:

```bash
python -m kaggle competitions download -c ucsc-cse-144-spring-2026-final-project
```

Extract and place as:
```
data/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ... (100 classes)
└── test/
    ├── 0.jpg
    └── ... (1000 images)
```

---

## Results

| Metric | Value |
|--------|-------|
| Best validation accuracy | 60.0% |
| Kaggle public leaderboard score | 62.727% |
| Backbone | EfficientNet-B2 |
| Epochs | 30 |
| Optimizer | AdamW (lr=1e-3) |
| Hardware | Google Colab T4 GPU |
