# Sketch-to-Image Generation with Custom Deep Learning Model

A custom deep learning project that transforms basic sketches into beautiful, detailed images using a Generative Adversarial Network (GAN) architecture.

## Features

- Custom U-Net based generator architecture
- PatchGAN discriminator for adversarial training
- End-to-end training pipeline
- Inference script for generating images from sketches
- Support for custom datasets

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── generator.py      # Custom generator model
│   └── discriminator.py  # Custom discriminator model
├── data/
│   └── dataset.py        # Data loading utilities
├── utils/
│   ├── __init__.py
│   └── utils.py          # Helper functions
├── train.py              # Training script
├── inference.py          # Inference script
├── config.py             # Configuration settings
└── requirements.txt      # Dependencies

```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

**Fastest way to get started:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python generate_training_data.py --num_samples 100

# 3. Train the model (start with fewer epochs for testing)
python train.py --data_dir ./dataset --epochs 20 --batch_size 4

# 4. Generate an image from a sketch
python inference.py \
    --checkpoint checkpoints/generator_best.pth \
    --input_sketch dataset/train/sketches/sketch_0001.png \
    --output_path result.png
```

**For detailed instructions, see [HOW_TO_RUN.md](HOW_TO_RUN.md)**

## Usage

### Quick Start - Generate Synthetic Training Data

**Don't have data yet?** Generate synthetic training data to test the system:

```bash
# Generate 100 simple samples
python generate_training_data.py --num_samples 100

# Generate more complex scene-based samples
python generate_training_data.py --num_samples 200 --advanced

# Custom output directory
python generate_training_data.py --output_dir ./my_dataset --num_samples 150
```

This will create sketch-image pairs automatically so you can start training immediately!

### Dataset Preparation

**Want to use your own data?** Here's how to prepare sketch-image pairs:

1. **Create directory structure:**
```bash
python prepare_dataset.py --data_dir ./dataset --create_structure
```

2. **Organize your data:**
   - Put your **sketch images** (grayscale or color) in `dataset/train/sketches/`
   - Put corresponding **real images** (the target beautiful images) in `dataset/train/images/`
   - Make sure each sketch has a matching image (same order or same filename)
   - Supported formats: `.png`, `.jpg`, `.jpeg`

3. **Validate your dataset:**
```bash
python prepare_dataset.py --data_dir ./dataset --validate
```

**Example dataset structure:**
```
dataset/
├── train/
│   ├── sketches/
│   │   ├── sketch_001.png
│   │   ├── sketch_002.png
│   │   └── ...
│   └── images/
│       ├── image_001.png  (corresponds to sketch_001.png)
│       ├── image_002.png  (corresponds to sketch_002.png)
│       └── ...
└── val/  (optional, for validation)
    ├── sketches/
    └── images/
```

**Important Notes:**
- Sketches can be grayscale or color (will be converted to grayscale automatically)
- Images should be RGB color images
- The number of sketches must match the number of images
- Files are paired by alphabetical order, so name them consistently

### Training

Run training:
```bash
python train.py --data_dir ./dataset --epochs 100 --batch_size 4
```

### Inference

Generate images from sketches:
```bash
python inference.py --checkpoint checkpoints/generator_best.pth --input_sketch path/to/sketch.png --output_path result.png
```

## Model Architecture

- **Generator**: Custom U-Net architecture with skip connections for preserving sketch details
- **Discriminator**: PatchGAN discriminator that classifies image patches as real or fake

## License

MIT

