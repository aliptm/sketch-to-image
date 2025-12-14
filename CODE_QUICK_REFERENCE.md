# Quick Code Reference - At a Glance

## 📋 File-by-File Summary

### `config.py`
**What**: Configuration file with all settings  
**Contains**: Device, paths, model params, training params, loss weights  
**Used by**: All scripts that need settings

---

### `models/generator.py`
**What**: U-Net Generator model  
**Does**: Transforms sketch (1 channel) → image (3 channels RGB)  
**Key**: Encoder-decoder with skip connections, 53M parameters  
**Output**: 256x256 RGB image

---

### `models/discriminator.py`
**What**: PatchGAN Discriminator model  
**Does**: Evaluates if image is real or fake  
**Key**: Takes sketch + image, outputs probability map (30x30)  
**Output**: Real/fake probability for each patch

---

### `data/dataset.py`
**What**: Data loading for training  
**Does**: Loads sketch-image pairs, applies transformations  
**Key**: Normalizes to [-1, 1], resizes to 256x256  
**Returns**: PyTorch DataLoader

---

### `utils/utils.py`
**What**: Helper functions  
**Functions**:
- `save_checkpoint()` - Save model
- `load_checkpoint()` - Load model
- `save_sample_images()` - Visualize training progress
- `initialize_weights()` - Initialize model weights

---

### `train.py`
**What**: Main training script  
**Does**: 
1. Trains discriminator (real vs fake)
2. Trains generator (fool discriminator + match structure)
3. Saves checkpoints
4. Logs to TensorBoard
**Run**: `python train.py --data_dir ./dataset --epochs 50`

---

### `inference.py`
**What**: Generate images from sketches  
**Does**:
1. Loads trained model
2. Processes input sketch
3. Generates image
4. Saves result
**Run**: `python inference.py --checkpoint model.pth --input_sketch sketch.png --output_path result.png`

---

### `generate_training_data.py`
**What**: Creates synthetic dataset  
**Does**: Generates sketch-image pairs automatically  
**Run**: `python generate_training_data.py --num_samples 100`

---

### `prepare_dataset.py`
**What**: Validates dataset  
**Does**: Checks structure, validates images, verifies pairs  
**Run**: `python prepare_dataset.py --data_dir ./dataset --validate`

---

### `test_models.py`
**What**: Tests model architectures  
**Does**: Verifies models work, checks shapes, counts parameters  
**Run**: `python test_models.py`

---

## 🔄 Data Flow

### Training:
```
Dataset → DataLoader → Generator → Discriminator → Loss → Backprop
```

### Inference:
```
Sketch → Preprocess → Generator → Postprocess → Image
```

---

## 🎯 Key Functions to Know

| Function | File | Purpose |
|----------|------|---------|
| `Generator.forward()` | generator.py | Generate image from sketch |
| `Discriminator.forward()` | discriminator.py | Evaluate real/fake |
| `SketchImageDataset.__getitem__()` | dataset.py | Load one sample |
| `train_discriminator()` | train.py | Train discriminator |
| `train_generator()` | train.py | Train generator |
| `generate_image()` | inference.py | Generate from sketch |

---

## 💡 Important Concepts

- **Normalization**: Images in [-1, 1] range
- **Skip Connections**: Preserve sketch details
- **Adversarial Loss**: Makes images realistic
- **L1 Loss**: Matches input structure
- **Checkpointing**: Save/load model state

---

For detailed explanations, see `CODE_EXPLANATION.md`

