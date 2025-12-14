"""
Utility functions for training and inference
"""
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)
    print(f"Checkpoint loaded: {filepath}")
    return epoch, loss


def save_sample_images(generator, sketch, real_image, epoch, save_dir, device):
    """Save sample images during training"""
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        fake_image = generator(sketch.to(device))
    
    # Convert tensors to numpy arrays
    sketch_np = sketch[0].cpu().numpy().transpose(1, 2, 0)
    if sketch_np.shape[2] == 1:
        sketch_np = sketch_np.squeeze(2)
    
    real_np = real_image[0].cpu().numpy().transpose(1, 2, 0)
    real_np = (real_np + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    real_np = np.clip(real_np, 0, 1)
    
    fake_np = fake_image[0].cpu().numpy().transpose(1, 2, 0)
    fake_np = (fake_np + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    fake_np = np.clip(fake_np, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sketch_np, cmap='gray' if len(sketch_np.shape) == 2 else None)
    axes[0].set_title('Input Sketch')
    axes[0].axis('off')
    
    axes[1].imshow(fake_np)
    axes[1].set_title('Generated Image')
    axes[1].axis('off')
    
    axes[2].imshow(real_np)
    axes[2].set_title('Real Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_epoch_{epoch}.png'))
    plt.close()
    
    generator.train()


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

