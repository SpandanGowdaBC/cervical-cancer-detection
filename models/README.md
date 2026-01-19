# Models Directory

Trained model files will be saved here.

## Model Files

After training, you'll have:
- `cnn_model.h5` - Trained CNN model
- `vit_model.pt` - Trained Vision Transformer
- `efficientnet_model.keras` - Trained EfficientNet
- `best_model.keras` - Best performing model

## Note

Model files are large and excluded from Git (see `.gitignore`).

## Loading Models

### TensorFlow/Keras Models

```python
from tensorflow import keras

model = keras.models.load_model('models/efficientnet_model.keras')
```

### PyTorch Models

```python
import torch

model = torch.load('models/vit_model.pt')
model.eval()
```

## Model Sizes

- CNN: ~60 MB
- Vision Transformer: ~340 MB
- EfficientNet: ~20 MB
