# 🚀 Quick Start Guide

This guide will help you get started with the Cervical Cancer Detection project quickly.

## Prerequisites

- Python 3.8 or higher
- GPU with CUDA support (recommended)
- 8GB+ RAM
- 10GB+ free disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cervical-cancer-detection.git
cd cervical-cancer-detection
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup

### Option 1: Download Herlev Dataset

1. Download the Herlev Pap Smear dataset from:
   - [Official Source](http://mde-lab.aegean.gr/index.php/downloads)
   - Or search "Herlev Pap Smear" on Kaggle

2. Extract to `data/raw/` directory

### Option 2: Use Your Own Dataset

Organize your dataset in the following structure:

```
data/raw/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Data Preparation

### 1. Augment the Dataset

```bash
python scripts/augment_data.py --input data/raw --output data/augmented --target 1794
```

This will:
- Apply data augmentation
- Balance classes to 1,794 images each
- Save augmented images to `data/augmented/`

### 2. Split the Dataset

```bash
python scripts/split_data.py --input data/augmented --output data/split_data --ratio 0.7 0.1 0.2
```

This creates:
- `data/split_data/train/` (70%)
- `data/split_data/val/` (10%)
- `data/split_data/test/` (20%)

## Training Models

### Train CNN Model

```bash
python train_cnn.py --data data/split_data --epochs 50 --batch-size 32
```

**Expected Output:**
- Training time: ~2-3 hours (GPU)
- Test accuracy: ~95.5%
- Model saved to: `models/cnn_model.h5`

### Train Vision Transformer

```bash
python train_vit.py --data data/split_data --epochs 5 --batch-size 16
```

**Expected Output:**
- Training time: ~4-5 hours (GPU)
- Test accuracy: ~96.2%
- Model saved to: `models/vit_model.pt`

### Train EfficientNet

```bash
python train_efficientnet.py --data data/split_data --epochs 50 --batch-size 32
```

**Expected Output:**
- Training time: ~3-4 hours (GPU)
- Test accuracy: ~96.5%
- Model saved to: `models/efficientnet_model.keras`

## Making Predictions

### Single Image Prediction

```bash
python predict.py --model models/efficientnet_model.keras --image path/to/image.jpg
```

### Batch Prediction

```bash
python predict.py --model models/efficientnet_model.keras --input-dir path/to/images/ --output results.csv
```

## Visualizing Results

### GradCAM Visualization

```bash
python visualize_gradcam.py --model models/vit_model.pt --image path/to/image.jpg --model-type vit
```

### View Training History

Results are automatically saved to `results/` directory:
- Training curves: `results/plots/`
- Confusion matrices: `results/confusion_matrices/`
- Classification reports: `results/classification_reports/`

## Using Jupyter Notebooks

### Start Jupyter

```bash
jupyter notebook
```

### Open the Main Notebook

Navigate to `notebooks/cervicalcancerdetectionmcaproject.ipynb`

This notebook contains:
- Complete pipeline from data loading to evaluation
- All three models
- GradCAM examples
- Visualization code

## Quick Test

To verify everything is working:

```bash
python -c "import tensorflow as tf; import torch; print('TensorFlow:', tf.__version__); print('PyTorch:', torch.__version__); print('GPU Available (TF):', tf.config.list_physical_devices('GPU')); print('GPU Available (PyTorch):', torch.cuda.is_available())"
```

Expected output:
```
TensorFlow: 2.x.x
PyTorch: 1.x.x
GPU Available (TF): [PhysicalDevice(...)]
GPU Available (PyTorch): True
```

## Common Issues

### Issue 1: CUDA Out of Memory

**Solution:** Reduce batch size
```bash
python train_cnn.py --batch-size 16  # Instead of 32
```

### Issue 2: Module Not Found

**Solution:** Ensure virtual environment is activated and dependencies installed
```bash
pip install -r requirements.txt
```

### Issue 3: Dataset Not Found

**Solution:** Check data directory structure
```bash
python -c "import os; print(os.listdir('data/split_data/train'))"
```

## Next Steps

1. **Experiment with Hyperparameters**: Edit `config.py`
2. **Try Different Augmentations**: Modify augmentation settings
3. **Ensemble Models**: Combine predictions from all three models
4. **Deploy**: Create a web interface or API

## Getting Help

- **Issues**: Open an issue on GitHub
- **Documentation**: See `README.md` and `MODELS.md`
- **Examples**: Check `notebooks/` directory

## Performance Benchmarks

| Hardware | Model | Training Time | Inference Time |
|----------|-------|---------------|----------------|
| RTX 3080 | CNN | 2h | 15ms |
| RTX 3080 | ViT | 4h | 45ms |
| RTX 3080 | EfficientNet | 3h | 20ms |
| CPU (i7) | CNN | 12h | 150ms |
| CPU (i7) | ViT | 24h | 500ms |
| CPU (i7) | EfficientNet | 15h | 200ms |

---

**🎉 You're all set! Happy training!**

For detailed documentation, see:
- [README.md](README.md) - Project overview
- [MODELS.md](MODELS.md) - Model architectures
- [DATASET.md](DATASET.md) - Dataset information
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
