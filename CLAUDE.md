# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image binary classification system (normal vs disease) using metric learning with PyTorch. The project achieved **0.93 accuracy** through a dual-model voting ensemble approach.

**Key insight**: Metric learning (Triplet Loss + KNN) significantly outperforms direct classification (CrossEntropy) for small medical image datasets.

## Running the Models

```bash
# Train the metric learning model (produces 0.92 accuracy)
cd ensemble_learning/
python metric_ensemble.py

# Generate final predictions using dual-model voting (0.93 accuracy)
python vote_ensemble.py

# Analyze model predictions and complementarity
python analyze_predictions.py

# Train baseline model (0.90 accuracy)
cd best_0.90/
python train_robust_pro.py
```

## Dependencies

```bash
pip install torch timm scikit-learn pandas numpy opencv-python matplotlib seaborn
```

**Hardware requirements**: CUDA GPU with 24GB+ VRAM, batch size 24 at 448×448 resolution.

## Architecture

### Data Flow

```
Medical Images (448×448)
    ↓
Dual Backbone Feature Extraction
├── ConvNeXt Large → 1,536 features
└── ViT Large → 1,024 features
    ↓
Concatenate → 2,560 features
    ↓
Adapter Network (2,560 → 512 → 128)
    ↓
L2 Normalized Embedding
    ↓
Triplet Loss Training (margin=0.5)
    ↓
KNN Classification (Auto-K search)
```

### Key Files

| File | Purpose | Accuracy |
|------|---------|----------|
| `ensemble_learning/vote_ensemble.py` | Dual-model soft voting (final) | 0.93 |
| `ensemble_learning/metric_ensemble.py` | Metric learning with Triplet Loss | 0.92 |
| `best_0.90/train_robust_pro.py` | Baseline model | 0.90 |
| `ensemble_learning/analyze_predictions.py` | Model comparison analysis | - |

### Model Weights

- `ensemble_learning/metric_checkpoints/best_metric_ensemble.pth` (~1.9GB)
- `best_0.90/robust_pro_model.pth` (~2GB)

## Critical Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| IMG_SIZE | 448 | Higher resolution crucial for medical image details |
| LR_HEAD | 1e-3 | Adapter learns quickly for task adaptation |
| LR_BACKBONE | 5e-6 | Preserves pretrained knowledge |
| MARGIN | 0.5 | Triplet Loss separation margin |
| EPOCHS | 20-25 | Sufficient for convergence |

Differential learning rates are essential: backbone uses 5e-6 to preserve pretrained features, adapter uses 1e-3 for quick task adaptation.

## Data Structure

```
data/
├── TrainSet/
│   ├── disease/    # 646 images (37%)
│   └── normal/     # 993 images (63%)
└── TestSet/        # 250 unlabeled images
```

Labels: 0=normal, 1=disease

## What Works / What Doesn't

**Effective approaches**:
- ConvNeXt + ViT backbone combination (CNN + Transformer complementarity)
- Metric learning over direct classification for small datasets
- High resolution (448 vs 384)
- Simple soft voting between high-accuracy models
- Test-Time Augmentation (4 views)

**Failed experiments**:
- Swin + EfficientNetV2 (0.74-0.79)
- ConvNeXtV2 + BEiT (0.71, VRAM issues forcing lower resolution)
- Adding low-accuracy models to ensemble (adds noise, not complementarity)
- Complex ensemble methods (stacking, blending)
- CrossEntropy direct classification (0.57)

## Ensemble Strategy

Only combine models with:
1. High individual accuracy (both >0.90)
2. True complementarity on critical samples
3. Models disagreeing on <5% of samples

The 0.90 and 0.92 models have 96.4% agreement but differ meaningfully on 9 samples (3.6%), enabling soft voting to achieve 0.93.

## Output Format

Predictions saved as CSV:
```csv
image_id,label
1,0
2,1
...
```

Final submission: `ensemble_learning/0.93.csv`
