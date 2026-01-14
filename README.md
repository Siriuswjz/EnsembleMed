# Medical Image Binary Classification

Metric learning approach for medical image classification using Triplet Loss + KNN.

**Final accuracy: 0.93**

## Structure

```
├── ensemble_learning/
│   ├── train.py         # Train metric learning model (0.92)
│   ├── base_train.py    # Baseline model (0.90)
│   ├── ensemble.py      # Dual-model soft voting (0.93)
│   └── analyze.py       # Compare model predictions
├── data/
│   ├── TrainSet/        # 1639 images (normal/disease)
│   └── TestSet/         # 250 test images
└── report.tex           # Technical report
```

## Usage

```bash
# Train baseline model (0.90)
cd ensemble_learning
python base_train.py

# Train metric model (0.92)
python train.py

# Generate final predictions (0.93)
python ensemble.py
```

## Key Techniques

- **Dual backbone**: ConvNeXt Large + ViT Large
- **Triplet Loss**: Metric learning for feature extraction
- **Differential LR**: 1e-3 for adapter, 5e-6 for backbone
- **High resolution**: 448x448
- **TTA**: 4-view test-time augmentation
- **Soft voting**: Weighted ensemble of 0.90 + 0.92 models

## Dependencies

```bash
pip install torch timm scikit-learn pandas numpy opencv-python
```

Hardware: CUDA GPU with 24GB+ VRAM
