"""
Data loading utilities for cervical cancer detection.
Handles dataset loading, preprocessing, and augmentation.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from config import *


def create_data_generators(train_dir, val_dir, test_dir, 
                           image_size=IMAGE_SIZE, 
                           batch_size=BATCH_SIZE,
                           augment=True):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        image_size: Target image size (default: 224)
        batch_size: Batch size (default: 32)
        augment: Whether to apply augmentation to training data
    
    Returns:
        train_generator, val_generator, test_generator
    """
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=AUGMENTATION_CONFIG['shear_range'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            fill_mode=AUGMENTATION_CONFIG['fill_mode']
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def create_tf_datasets(train_dir, val_dir, test_dir,
                       image_size=IMAGE_SIZE,
                       batch_size=BATCH_SIZE):
    """
    Create TensorFlow datasets for EfficientNet and other models.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        image_size: Target image size
        batch_size: Batch size
    
    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    
    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names


def create_pytorch_dataloaders(train_dir, val_dir, test_dir,
                               transform,
                               batch_size=16):
    """
    Create PyTorch DataLoaders for Vision Transformer.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        transform: Torchvision transforms
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed image as numpy array
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        train_generator: Training data generator
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    return dict(enumerate(class_weights))


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading utilities...")
    
    train_dir = os.path.join(SPLIT_DATA_DIR, 'train')
    val_dir = os.path.join(SPLIT_DATA_DIR, 'val')
    test_dir = os.path.join(SPLIT_DATA_DIR, 'test')
    
    if os.path.exists(train_dir):
        train_gen, val_gen, test_gen = create_data_generators(
            train_dir, val_dir, test_dir
        )
        print(f"✅ Data generators created successfully")
        print(f"   Training samples: {train_gen.samples}")
        print(f"   Validation samples: {val_gen.samples}")
        print(f"   Test samples: {test_gen.samples}")
        print(f"   Classes: {train_gen.class_indices}")
    else:
        print(f"❌ Data directory not found: {train_dir}")
