# 📓 Notebooks

This directory contains Jupyter notebooks and Python scripts for the project.

## Files

### `cervicalcancerdetectionmcaproject.py`

The original implementation file containing:
- Complete data pipeline (extraction, augmentation, splitting)
- Three model implementations (CNN, Vision Transformer, EfficientNet)
- Training and evaluation code
- GradCAM visualization examples

**Note**: This is a Python script exported from Google Colab. To use it as a Jupyter notebook:

1. Convert to .ipynb format:
   ```bash
   jupytext --to notebook cervicalcancerdetectionmcaproject.py
   ```

2. Or run directly in Colab:
   - Upload to Google Colab
   - Run cells sequentially

## Usage

### Running in Jupyter

```bash
jupyter notebook cervicalcancerdetectionmcaproject.py
```

### Running in Google Colab

1. Upload the file to Google Colab
2. Mount Google Drive (if using Drive for dataset)
3. Run cells in order

## Contents Overview

1. **Data Preparation**
   - Google Drive mounting
   - Dataset extraction
   - Data augmentation (13x per image)
   - Class balancing (1,794 per class)
   - Train/Val/Test split (70/10/20)

2. **Model Training**
   - CNN Model
   - Vision Transformer (ViT)
   - EfficientNetB0

3. **Evaluation**
   - Classification reports
   - Confusion matrices
   - Training curves

4. **GradCAM Visualization**
   - CNN GradCAM
   - ViT GradCAM
   - Multiple examples

## Creating New Notebooks

When creating new notebooks:
1. Use descriptive names
2. Add markdown cells for documentation
3. Include code comments
4. Save outputs for reference

---

*For more information, see the main [README.md](../README.md)*
