"""
Inference script for generating images from sketches
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

from models import Generator
from config import Config
from utils.utils import load_checkpoint


def load_image(image_path, image_size=256, is_sketch=True):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    
    if is_sketch:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def save_image(tensor, output_path):
    """Save tensor as image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.squeeze(0).cpu())
    image.save(output_path)
    print(f"Generated image saved to: {output_path}")


def generate_image(sketch_path, checkpoint_path, output_path, config):
    """Generate image from sketch"""
    # Setup device
    device = config.device
    print(f"Using device: {device}")
    
    # Load model
    generator = Generator(
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        features=config.generator_features
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(checkpoint_path, generator, device=device)
    generator.eval()
    
    # Load sketch
    print(f"Loading sketch: {sketch_path}")
    sketch = load_image(sketch_path, config.image_size, is_sketch=True)
    sketch = sketch.to(device)
    
    # Generate image
    print("Generating image...")
    with torch.no_grad():
        generated_image = generator(sketch)
    
    # Save result
    save_image(generated_image, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image from sketch')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--input_sketch', type=str, required=True,
                       help='Path to input sketch image')
    parser.add_argument('--output_path', type=str, default='./generated_image.png',
                       help='Path to save generated image')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.image_size = args.image_size
    config.checkpoint_path = args.checkpoint
    
    # Generate image
    generate_image(
        args.input_sketch,
        args.checkpoint,
        args.output_path,
        config
    )

