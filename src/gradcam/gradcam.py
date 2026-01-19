"""
GradCAM implementation for model interpretability.
Supports CNN, Vision Transformer, and EfficientNet models.
"""

import numpy as np
import cv2
import torch
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """GradCAM for TensorFlow/Keras models (CNN, EfficientNet)"""
    
    def __init__(self, model, layer_name):
        """
        Initialize GradCAM.
        
        Args:
            model: Keras model
            layer_name: Name of the target convolutional layer
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        """Build gradient model for computing GradCAM."""
        return keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
    
    def generate_heatmap(self, image, class_idx=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            image: Input image (preprocessed)
            class_idx: Target class index (None for predicted class)
        
        Returns:
            Heatmap as numpy array
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_output = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: GradCAM heatmap
            image: Original image
            alpha: Transparency of heatmap
            colormap: OpenCV colormap
        
        Returns:
            Superimposed image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize image
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Superimpose
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed


class ViTGradCAM:
    """GradCAM for Vision Transformer models"""
    
    def __init__(self, model, device):
        """
        Initialize ViT GradCAM.
        
        Args:
            model: Vision Transformer model
            device: torch device
        """
        self.model = model
        self.device = device
        self.activations = []
        self.gradients = []
    
    def _normalize_cam(self, cam):
        """Normalize CAM values."""
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-5)
        return cam
    
    def generate_heatmap(self, image_tensor):
        """
        Generate GradCAM heatmap for Vision Transformer.
        
        Args:
            image_tensor: Input image tensor
        
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        self.activations = []
        self.gradients = []
        
        def forward_hook(module, input, output):
            self.activations.append(output)
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])
        
        # Hook into last encoder block
        target_layer = self.model.vit.encoder.layer[-1].output
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        outputs = self.model(pixel_values=image_tensor).logits
        pred_class = outputs.argmax(dim=1)
        loss = outputs[:, pred_class]
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Compute CAM
        grads = self.gradients[0]
        acts = self.activations[0]
        
        weights = grads.mean(dim=-1)
        cam = (weights.unsqueeze(-1) * acts).sum(dim=1)
        
        cam = cam.squeeze(0)
        cam = self._normalize_cam(cam.cpu().detach().numpy())
        
        # Remove CLS token
        cam = cam[1:]
        
        # Reshape to 2D
        num_patches = cam.shape[0]
        side = int(np.ceil(np.sqrt(num_patches)))
        pad_len = (side * side) - num_patches
        cam = np.pad(cam, (0, pad_len), mode='constant')
        cam = cam.reshape(side, side)
        
        # Resize to image size
        cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().numpy()
        
        # Remove hooks
        fwd_handle.remove()
        bwd_handle.remove()
        
        return cam
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: GradCAM heatmap
            image: Original PIL Image
            alpha: Transparency
        
        Returns:
            Superimposed image
        """
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Convert image to array
        img_np = np.array(image) / 255
        
        # Superimpose
        superimposed = heatmap + img_np
        superimposed = superimposed / np.max(superimposed)
        
        return superimposed


def visualize_gradcam(image_path, model, layer_name=None, model_type='cnn', 
                     device=None, transform=None, save_path=None):
    """
    Unified function to visualize GradCAM for any model.
    
    Args:
        image_path: Path to input image
        model: Trained model
        layer_name: Target layer name (for CNN/EfficientNet)
        model_type: 'cnn', 'efficientnet', or 'vit'
        device: torch device (for ViT)
        transform: Image transform (for ViT)
        save_path: Optional path to save visualization
    """
    if model_type in ['cnn', 'efficientnet']:
        # Load and preprocess image
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0
        
        # Generate GradCAM
        gradcam = GradCAM(model, layer_name)
        heatmap = gradcam.generate_heatmap(img_array_expanded)
        superimposed = gradcam.overlay_heatmap(heatmap, img_array / 255.0)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Superimposed', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
    elif model_type == 'vit':
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        
        # Generate GradCAM
        gradcam = ViTGradCAM(model, device)
        heatmap = gradcam.generate_heatmap(img_tensor)
        superimposed = gradcam.overlay_heatmap(heatmap, img)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Superimposed', fontsize=12, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ GradCAM visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("GradCAM utilities loaded successfully!")
