# How to Run the Sketch-to-Image Generation Project

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (optional, but recommended for training)

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- torchvision (image processing)
- numpy, Pillow (image handling)
- matplotlib (visualization)
- tqdm (progress bars)
- tensorboard (training monitoring)

---

## Step 2: Prepare Your Dataset

### Option A: Generate Synthetic Data (Quick Start)

If you don't have your own data yet, generate synthetic training data:

```bash
# Generate 100 simple samples
python generate_training_data.py --num_samples 100

# Or generate more complex scene-based samples
python generate_training_data.py --num_samples 200 --advanced

# Custom output directory
python generate_training_data.py --output_dir ./my_dataset --num_samples 150
```

This creates:
- `dataset/train/sketches/` - Sketch images
- `dataset/train/images/` - Corresponding beautiful images

### Option B: Use Your Own Data

1. Create directory structure:
```bash
python prepare_dataset.py --data_dir ./dataset --create_structure
```

2. Organize your files:
   - Put sketch images in `dataset/train/sketches/`
   - Put corresponding real images in `dataset/train/images/`
   - Make sure filenames match or are in the same order

3. Validate your dataset:
```bash
python prepare_dataset.py --data_dir ./dataset --validate
```

---

## Step 3: Train the Model

### Basic Training

```bash
python train.py --data_dir ./dataset --epochs 50 --batch_size 4
```

### Training with Custom Parameters

```bash
python train.py \
    --data_dir ./dataset \
    --epochs 100 \
    --batch_size 4 \
    --image_size 256 \
    --lr 0.0002
```

### Training Parameters Explained

- `--data_dir`: Path to your dataset directory
- `--epochs`: Number of training epochs (50-100 recommended)
- `--batch_size`: Batch size (4-8 depending on GPU memory)
- `--image_size`: Image size (256 is default)
- `--lr`: Learning rate (0.0002 is default)

### Monitor Training

Training automatically saves:
- **Checkpoints**: `checkpoints/checkpoint_epoch_X.pth`
- **Best model**: `checkpoints/generator_best.pth`
- **Sample images**: `logs/samples/sample_epoch_X.png`
- **TensorBoard logs**: `logs/`

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```
Then open http://localhost:6006 in your browser

---

## Step 4: Generate Images from Sketches (Inference)

After training, generate images from new sketches:

```bash
python inference.py \
    --checkpoint checkpoints/generator_best.pth \
    --input_sketch path/to/your/sketch.png \
    --output_path result.png
```

### Example

```bash
# Use the best trained model
python inference.py \
    --checkpoint checkpoints/generator_best.pth \
    --input_sketch dataset/train/sketches/sketch_0001.png \
    --output_path generated_image.png
```

---

## Complete Workflow Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (or use your own)
python generate_training_data.py --num_samples 100

# 3. Validate dataset
python prepare_dataset.py --data_dir ./dataset --validate

# 4. Test models work
python test_models.py

# 5. Train the model
python train.py --data_dir ./dataset --epochs 50 --batch_size 4

# 6. Generate images
python inference.py \
    --checkpoint checkpoints/generator_best.pth \
    --input_sketch dataset/train/sketches/sketch_0001.png \
    --output_path my_result.png
```

---

## Troubleshooting

### Out of Memory Error
- Reduce `--batch_size` to 2 or 1
- Reduce `--image_size` to 128

### Training is Slow
- Use GPU if available (CUDA)
- Reduce number of epochs for testing
- Use smaller batch size

### Model Not Learning
- Check dataset quality (run validation)
- Increase number of epochs
- Adjust learning rate (try 0.0001 or 0.0005)

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

---

## Project Structure

```
.
├── models/
│   ├── generator.py      # Generator model
│   └── discriminator.py  # Discriminator model
├── data/
│   └── dataset.py        # Data loading
├── utils/
│   └── utils.py         # Helper functions
├── train.py             # Training script
├── inference.py         # Inference script
├── generate_training_data.py  # Data generation
├── prepare_dataset.py    # Dataset validation
├── test_models.py       # Test models
├── config.py            # Configuration
└── dataset/             # Your data
    └── train/
        ├── sketches/
        └── images/
```

---

## Quick Commands Reference

```bash
# Generate data
python generate_training_data.py --num_samples 100

# Validate dataset
python prepare_dataset.py --data_dir ./dataset --validate

# Test models
python test_models.py

# Train
python train.py --data_dir ./dataset --epochs 50

# Generate image
python inference.py --checkpoint checkpoints/generator_best.pth --input_sketch sketch.png --output_path result.png

# View training
tensorboard --logdir logs
```

---

## Expected Training Time

- **CPU**: ~2-4 hours per epoch (not recommended)
- **GPU (NVIDIA)**: ~5-15 minutes per epoch
- **Total**: 50 epochs = ~4-12 hours on GPU

For testing, you can train for fewer epochs (10-20) to see results faster.

---

## Next Steps After Training

1. **Evaluate results**: Check `logs/samples/` for generated images
2. **Fine-tune**: Adjust hyperparameters if needed
3. **Generate more**: Use inference script on new sketches
4. **Improve**: Add more training data, adjust architecture

---

## Need Help?

- Check `README.md` for project overview
- Review `INTERVIEW_GUIDE.md` for project explanation
- Look at code comments in model files
- Test with `test_models.py` to verify setup

