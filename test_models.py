"""
Test script to verify model architectures
"""
import torch
from models import Generator, Discriminator

def test_models():
    """Test generator and discriminator models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")
    
    # Test Generator
    print("=" * 50)
    print("Testing Generator")
    print("=" * 50)
    generator = Generator(input_channels=1, output_channels=3, features=64)
    generator = generator.to(device)
    
    sketch = torch.randn(2, 1, 256, 256).to(device)
    print(f"Input sketch shape: {sketch.shape}")
    
    with torch.no_grad():
        generated = generator(sketch)
    
    print(f"Generated image shape: {generated.shape}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Generator size: {sum(p.numel() * p.element_size() for p in generator.parameters()) / 1024 / 1024:.2f} MB\n")
    
    # Test Discriminator
    print("=" * 50)
    print("Testing Discriminator")
    print("=" * 50)
    discriminator = Discriminator(input_channels=1, image_channels=3, features=64)
    discriminator = discriminator.to(device)
    
    image = torch.randn(2, 3, 256, 256).to(device)
    print(f"Input sketch shape: {sketch.shape}")
    print(f"Input image shape: {image.shape}")
    
    with torch.no_grad():
        output = discriminator(sketch, image)
    
    print(f"Discriminator output shape: {output.shape}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Discriminator size: {sum(p.numel() * p.element_size() for p in discriminator.parameters()) / 1024 / 1024:.2f} MB\n")
    
    # Test forward pass together
    print("=" * 50)
    print("Testing Full Forward Pass")
    print("=" * 50)
    generator.train()
    discriminator.train()
    
    fake_image = generator(sketch)
    fake_pred = discriminator(sketch, fake_image)
    real_pred = discriminator(sketch, image)
    
    print(f"Fake image prediction shape: {fake_pred.shape}")
    print(f"Real image prediction shape: {real_pred.shape}")
    print(f"Fake prediction mean: {fake_pred.mean().item():.4f}")
    print(f"Real prediction mean: {real_pred.mean().item():.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed! [SUCCESS]")
    print("=" * 50)

if __name__ == "__main__":
    test_models()

