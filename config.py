"""
Configuration file for the Cervical Cancer Detection project.
Contains all hyperparameters and settings for training and inference.
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, 'augmented')
SPLIT_DATA_DIR = os.path.join(DATA_DIR, 'split_data')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATA SETTINGS ====================
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5  # Adjust based on your dataset

# Class names (Herlev dataset)
CLASS_NAMES = [
    'carcinoma_in_situ',
    'light_dysplastic',
    'moderate_dysplastic',
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
    'severe_dysplastic'
]

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ==================== AUGMENTATION SETTINGS ====================
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

AUGMENTATIONS_PER_IMAGE = 13
TARGET_IMAGES_PER_CLASS = 1794

# ==================== CNN MODEL SETTINGS ====================
CNN_CONFIG = {
    'model_name': 'cnn_model',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'dropout_rate': 0.5,
    'early_stopping_patience': 8,
    'reduce_lr_patience': 4,
    'reduce_lr_factor': 0.5
}

# ==================== VISION TRANSFORMER SETTINGS ====================
VIT_CONFIG = {
    'model_name': 'vit_model',
    'pretrained_model': 'google/vit-base-patch16-224-in21k',
    'epochs': 5,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01
}

# ==================== EFFICIENTNET SETTINGS ====================
EFFICIENTNET_CONFIG = {
    'model_name': 'efficientnet_model',
    'base_model': 'EfficientNetB0',
    'epochs': 50,
    'fine_tune_epochs': 20,
    'batch_size': 32,
    'initial_learning_rate': 1e-4,
    'fine_tune_learning_rate': 1e-5,
    'dropout_rate': 0.5,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.2,
    'fine_tune_at': 0.75  # Freeze first 75% of layers
}

# ==================== TRAINING SETTINGS ====================
RANDOM_SEED = 123
USE_GPU = True
MIXED_PRECISION = True  # For faster training on compatible GPUs

# ==================== GRADCAM SETTINGS ====================
GRADCAM_CONFIG = {
    'colormap': 'jet',
    'alpha': 0.4,  # Transparency of heatmap overlay
    'interpolation': 'bilinear'
}

# ==================== EVALUATION SETTINGS ====================
METRICS = ['accuracy', 'precision', 'recall', 'f1-score']
SAVE_CONFUSION_MATRIX = True
SAVE_CLASSIFICATION_REPORT = True
SAVE_TRAINING_PLOTS = True

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'
TENSORBOARD_LOG_DIR = os.path.join(LOGS_DIR, 'tensorboard')
