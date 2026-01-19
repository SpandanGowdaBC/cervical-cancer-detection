# Data Directory

Place your dataset files here.

## Structure

```
data/
├── raw/              # Original, unprocessed dataset
├── augmented/        # Dataset after augmentation
└── split_data/       # Train/Val/Test split
    ├── train/
    ├── val/
    └── test/
```

## Getting the Dataset

### Herlev Pap Smear Dataset

Download from:
- [Official Source](http://mde-lab.aegean.gr/index.php/downloads)
- Kaggle: Search for "Herlev Pap Smear"

### Setup

1. Download the dataset ZIP file
2. Extract to `data/raw/`
3. Run augmentation script:
   ```bash
   python scripts/augment_data.py --input data/raw --output data/augmented
   ```
4. Split the data:
   ```bash
   python scripts/split_data.py --input data/augmented --output data/split_data
   ```

## Note

The `data/` directory is excluded from Git (see `.gitignore`) to avoid committing large files.
