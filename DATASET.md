# 📊 Dataset Information

## Herlev Pap Smear Dataset

### Overview

The Herlev Pap Smear dataset is a benchmark dataset for cervical cell classification, collected at Herlev University Hospital in Denmark. It contains single-cell images extracted from Pap smear slides.

### Dataset Statistics

#### Original Dataset
- **Total Images**: ~917 images
- **Image Format**: BMP
- **Resolution**: Variable (typically 224×224 after preprocessing)
- **Color Space**: RGB
- **Bit Depth**: 24-bit

#### Classes

The dataset contains **7 classes** representing different cell types:

| Class | Description | Original Count |
|-------|-------------|----------------|
| **Normal Superficial** | Healthy superficial squamous cells | ~74 |
| **Normal Intermediate** | Healthy intermediate squamous cells | ~70 |
| **Normal Columnar** | Healthy columnar epithelial cells | ~98 |
| **Light Dysplastic** | Mild abnormality (CIN 1) | ~182 |
| **Moderate Dysplastic** | Moderate abnormality (CIN 2) | ~146 |
| **Severe Dysplastic** | Severe abnormality (CIN 3) | ~197 |
| **Carcinoma in Situ** | Pre-cancerous cells | ~150 |

**CIN**: Cervical Intraepithelial Neoplasia

### Class Imbalance

The original dataset has class imbalance, which we address through:
1. Data augmentation
2. Class balancing
3. Weighted loss functions (optional)

---

## Data Augmentation Strategy

### Motivation

- **Increase Dataset Size**: From ~917 to 12,558 images (1,794 per class)
- **Balance Classes**: Ensure equal representation
- **Improve Generalization**: Reduce overfitting
- **Simulate Real-World Variations**: Account for imaging variations

### Augmentation Techniques

```python
ImageDataGenerator(
    rotation_range=30,           # Random rotation ±30°
    width_shift_range=0.2,       # Horizontal shift up to 20%
    height_shift_range=0.2,      # Vertical shift up to 20%
    shear_range=0.15,            # Shear transformation
    zoom_range=0.2,              # Random zoom 80-120%
    horizontal_flip=True,        # Random horizontal flip
    vertical_flip=True,          # Random vertical flip
    brightness_range=[0.8, 1.2], # Brightness adjustment
    fill_mode='nearest'          # Fill empty pixels
)
```

### Augmentation Process

1. **Initial Augmentation**: 13 augmented versions per original image
2. **Class Balancing**: Generate additional images for minority classes
3. **Target**: 1,794 images per class (balanced dataset)

### Augmented Dataset Statistics

| Class | Augmented Count | Increase |
|-------|----------------|----------|
| Normal Superficial | 1,794 | ~24× |
| Normal Intermediate | 1,794 | ~26× |
| Normal Columnar | 1,794 | ~18× |
| Light Dysplastic | 1,794 | ~10× |
| Moderate Dysplastic | 1,794 | ~12× |
| Severe Dysplastic | 1,794 | ~9× |
| Carcinoma in Situ | 1,794 | ~12× |

**Total Augmented Images**: 12,558

---

## Data Split

### Split Ratios

- **Training Set**: 70% (8,790 images)
- **Validation Set**: 10% (1,256 images)
- **Test Set**: 20% (2,512 images)

### Split Strategy

```python
splitfolders.ratio(
    input_folder,
    output="split_data",
    seed=123,              # For reproducibility
    ratio=(0.7, 0.1, 0.2),
    move=False             # Copy instead of move
)
```

### Per-Class Distribution

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Normal Superficial | 1,256 | 179 | 359 | 1,794 |
| Normal Intermediate | 1,256 | 179 | 359 | 1,794 |
| Normal Columnar | 1,256 | 179 | 359 | 1,794 |
| Light Dysplastic | 1,256 | 179 | 359 | 1,794 |
| Moderate Dysplastic | 1,256 | 179 | 359 | 1,794 |
| Severe Dysplastic | 1,256 | 179 | 359 | 1,794 |
| Carcinoma in Situ | 1,256 | 179 | 359 | 1,794 |

---

## Data Preprocessing

### Image Preprocessing Pipeline

1. **Loading**: Read image from disk
2. **Resizing**: Resize to 224×224 (model input size)
3. **Normalization**: 
   - CNN: Rescale to [0, 1]
   - EfficientNet: EfficientNet-specific preprocessing
   - ViT: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
4. **Augmentation**: Apply random transformations (training only)
5. **Batching**: Group into batches of 16-32 images

### Code Example

```python
# TensorFlow/Keras
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    # ... other augmentations
)

train_generator = train_datagen.flow_from_directory(
    'data/split_data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# PyTorch (for ViT)
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], 
              std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder('data/split_data/train', transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## Dataset Preparation Script

### Usage

```bash
# Extract dataset
python scripts/prepare_data.py --input data/Herlev_PapSmear.zip --output data/raw

# Augment data
python scripts/augment_data.py --input data/raw --output data/augmented --target 1794

# Split data
python scripts/split_data.py --input data/augmented --output data/split_data --ratio 0.7 0.1 0.2
```

---

## Data Quality Considerations

### Challenges

1. **Small Original Dataset**: Only ~917 images
   - **Solution**: Extensive augmentation
   
2. **Class Imbalance**: Varying class sizes
   - **Solution**: Balanced augmentation to 1,794 per class
   
3. **Single-Cell Images**: May not reflect real-world slides
   - **Solution**: Augmentation simulates variations
   
4. **Limited Diversity**: Single source (one hospital)
   - **Future Work**: Incorporate multi-center datasets

### Quality Assurance

- Manual inspection of augmented images
- Verification of class distribution
- Sanity checks on data splits
- Validation of preprocessing pipeline

---

## Citation

If you use the Herlev dataset, please cite:

```bibtex
@article{jantzen2005pap,
  title={Pap-smear benchmark data for pattern classification},
  author={Jantzen, Jan and Norup, Jonas and Dounias, Georgios and Bjerregaard, Beth},
  journal={Nature inspired Smart Information Systems (NiSIS 2005)},
  pages={1--9},
  year={2005}
}
```

---

## Download Links

**Official Dataset**: 
- [Herlev University Hospital](http://mde-lab.aegean.gr/index.php/downloads)

**Alternative Sources**:
- Kaggle: Search for "Herlev Pap Smear"
- UCI Machine Learning Repository

---

## Data Ethics and Privacy

- All images are anonymized
- No patient identifiable information
- Used for research and educational purposes
- Complies with medical data sharing regulations

---

*Last Updated: January 2026*
