"""
Visualization utilities for training results and model predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras History object or dict with 'accuracy', 'loss', etc.
        save_path: Optional path to save the plot
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to {save_path}")
    
    plt.show()


def plot_combined_history(histories, labels, save_path=None):
    """
    Plot multiple training histories (e.g., initial + fine-tuning).
    
    Args:
        histories: List of history objects
        labels: List of labels for each history
        save_path: Optional path to save the plot
    """
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    
    for h in histories:
        if hasattr(h, 'history'):
            h = h.history
        acc += h['accuracy']
        val_acc += h['val_accuracy']
        loss += h['loss']
        val_loss += h['val_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(acc, label='Training Accuracy', linewidth=2)
    axes[0].plot(val_acc, label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy (Combined)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(loss, label='Training Loss', linewidth=2)
    axes[1].plot(val_loss, label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss (Combined)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm


def print_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Print and optionally save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\n" + "="*60)
    print("📝 CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(report)
        print(f"✅ Classification report saved to {save_path}")
    
    return report


def plot_sample_predictions(model, test_generator, class_names, num_samples=9, save_path=None):
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
    """
    # Get a batch of test images
    x_batch, y_batch = next(test_generator)
    predictions = model.predict(x_batch)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        if i >= len(x_batch):
            break
        
        # Get true and predicted labels
        true_label = class_names[np.argmax(y_batch[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        
        # Plot image
        axes[i].imshow(x_batch[i])
        axes[i].axis('off')
        
        # Color: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                         fontsize=10, color=color, fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sample predictions saved to {save_path}")
    
    plt.show()


def plot_class_distribution(data_generator, title='Class Distribution', save_path=None):
    """
    Plot class distribution in the dataset.
    
    Args:
        data_generator: Data generator
        title: Plot title
        save_path: Optional path to save the plot
    """
    class_counts = np.bincount(data_generator.classes)
    class_names = list(data_generator.class_indices.keys())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_counts, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison of multiple models.
    
    Args:
        results_dict: Dictionary with model names as keys and accuracy as values
                     Example: {'CNN': 0.955, 'ViT': 0.962, 'EfficientNet': 0.965}
        save_path: Optional path to save the plot
    """
    models = list(results_dict.keys())
    accuracies = [results_dict[m] * 100 for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                   edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.ylim([90, 100])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
