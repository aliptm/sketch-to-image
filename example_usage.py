"""
Example usage script showing how to use the sketch-to-image generation system
"""
import os
import torch
from models import Generator, Discriminator
from config import Config

def example_model_creation():
    """Example: Create and test models"""
    print("Example 1: Creating Models")
    print("-" * 50)
    
    config = Config()
    device = config.device
    
    # Create generator
    generator = Generator(
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        features=config.generator_features
    ).to(device)
    
    # Create discriminator
    discriminator = Discriminator(
        input_channels=config.input_channels,
        image_channels=config.output_channels,
        features=config.discriminator_features
    ).to(device)
    
    print(f"Generator created with {sum(p.numel() for p in generator.parameters()):,} parameters")
    print(f"Discriminator created with {sum(p.numel() for p in discriminator.parameters()):,} parameters")
    print()


def example_training_command():
    """Example: Training command"""
    print("Example 2: Training Command")
    print("-" * 50)
    print("To train the model, use:")
    print()
    print("python train.py \\")
    print("    --data_dir ./dataset \\")
    print("    --epochs 100 \\")
    print("    --batch_size 4 \\")
    print("    --image_size 256 \\")
    print("    --lr 0.0002")
    print()
    print("Make sure your dataset structure is:")
    print("dataset/")
    print("  train/")
    print("    sketches/  (your sketch images)")
    print("    images/    (corresponding real images)")
    print("  val/")
    print("    sketches/")
    print("    images/")
    print()


def example_inference_command():
    """Example: Inference command"""
    print("Example 3: Inference Command")
    print("-" * 50)
    print("To generate an image from a sketch, use:")
    print()
    print("python inference.py \\")
    print("    --checkpoint checkpoints/generator_best.pth \\")
    print("    --input_sketch path/to/your/sketch.png \\")
    print("    --output_path generated_result.png")
    print()


def example_custom_training():
    """Example: Custom training loop"""
    print("Example 4: Custom Training Loop")
    print("-" * 50)
    print("""
# Example custom training code snippet:

from models import Generator, Discriminator
from data.dataset import get_dataloader
from config import Config
import torch.nn as nn
import torch.optim as optim

config = Config()
device = config.device

# Initialize models
generator = Generator(...).to(device)
discriminator = Discriminator(...).to(device)

# Loss functions
criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# Data loader
train_loader = get_dataloader(...)

# Training loop
for epoch in range(num_epochs):
    for sketch, real_image in train_loader:
        sketch = sketch.to(device)
        real_image = real_image.to(device)
        
        # Train discriminator
        # ... (see train.py for full implementation)
        
        # Train generator
        # ... (see train.py for full implementation)
""")


if __name__ == "__main__":
    print("=" * 60)
    print("Sketch-to-Image Generation - Usage Examples")
    print("=" * 60)
    print()
    
    example_model_creation()
    example_training_command()
    example_inference_command()
    example_custom_training()
    
    print("=" * 60)
    print("For more details, see README.md")
    print("=" * 60)

