"""
Custom Discriminator Model - PatchGAN Architecture
Classifies image patches as real or fake
"""
import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """Discriminator building block"""
    def __init__(self, in_channels, out_channels, stride=2, use_norm=True):
        super(DiscriminatorBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Custom PatchGAN Discriminator
    Takes sketch-image pairs and classifies patches as real or fake
    """
    def __init__(self, input_channels=1, image_channels=3, features=64):
        super(Discriminator, self).__init__()
        
        # Input: sketch (1 channel) + image (3 channels) = 4 channels
        self.model = nn.Sequential(
            # First layer without normalization
            nn.Conv2d(input_channels + image_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling layers
            DiscriminatorBlock(features, features * 2, stride=2, use_norm=True),
            DiscriminatorBlock(features * 2, features * 4, stride=2, use_norm=True),
            DiscriminatorBlock(features * 4, features * 8, stride=1, use_norm=True),
            
            # Final classification layer
            nn.Conv2d(features * 8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, sketch, image):
        # Concatenate sketch and image along channel dimension
        x = torch.cat([sketch, image], dim=1)
        return self.model(x)


def test_discriminator():
    """Test function to verify discriminator architecture"""
    model = Discriminator(input_channels=1, image_channels=3, features=64)
    sketch = torch.randn(1, 1, 256, 256)
    image = torch.randn(1, 3, 256, 256)
    output = model(sketch, image)
    print(f"Sketch shape: {sketch.shape}")
    print(f"Image shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Discriminator parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_discriminator()

