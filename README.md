# Medical Image Binary Classification

Metric learning approach for medical image classification using Triplet Loss + KNN.

**Final accuracy: 0.93**

## Structure

```
├── ensemble_learning/
│   ├── train.py         # Train metric learning model
│   ├── base_train.py    # Baseline model
│   ├── ensemble.py      # Dual-model soft voting
│   └── analyze.py       # Compare model predictions
├── data/
│   ├── TrainSet/        # 1639 images (normal/disease)
│   └── TestSet/         # 250 test images
└── report.tex           # Technical report
```

## Usage

```bash
# Train baseline model
cd ensemble_learning
python base_train.py

# Train metric model 
python train.py

# Generate final predictions 
python ensemble.py
```

## Key Techniques

- **Dual backbone**: ConvNeXt Large + ViT Large
- **Triplet Loss**: Metric learning for feature extraction
- **Differential LR**: 1e-3 for adapter, 5e-6 for backbone
- **High resolution**: 448x448
- **TTA**: 4-view test-time augmentation

## Dependencies

```bash
pip install torch timm scikit-learn pandas numpy opencv-python
```

Hardware: CUDA GPU with 24GB+ VRAM
