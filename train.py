"""
Training script for sketch-to-image generation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm

from models import Generator, Discriminator
from data.dataset import get_dataloader
from utils.utils import save_checkpoint, save_sample_images, initialize_weights
from config import Config


def train_discriminator(discriminator, generator, sketch, real_image, 
                       optimizer_d, criterion, device, config):
    """Train discriminator"""
    optimizer_d.zero_grad()
    
    # Real images
    real_pred = discriminator(sketch, real_image)
    real_loss = criterion(real_pred, torch.ones_like(real_pred))
    
    # Fake images
    fake_image = generator(sketch)
    fake_pred = discriminator(sketch, fake_image.detach())
    fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
    
    # Total discriminator loss
    d_loss = (real_loss + fake_loss) / 2
    
    d_loss.backward()
    optimizer_d.step()
    
    return d_loss.item()


def train_generator(generator, discriminator, sketch, real_image,
                   optimizer_g, criterion_gan, criterion_l1, device, config):
    """Train generator"""
    optimizer_g.zero_grad()
    
    # Generate fake image
    fake_image = generator(sketch)
    
    # Discriminator prediction on fake image
    fake_pred = discriminator(sketch, fake_image)
    gan_loss = criterion_gan(fake_pred, torch.ones_like(fake_pred))
    
    # L1 loss (pixel-wise difference)
    l1_loss = criterion_l1(fake_image, real_image)
    
    # Total generator loss
    g_loss = config.lambda_gan * gan_loss + config.lambda_l1 * l1_loss
    
    g_loss.backward()
    optimizer_g.step()
    
    return g_loss.item(), gan_loss.item(), l1_loss.item()


def train(config):
    """Main training function"""
    # Setup device
    device = config.device
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.log_dir, 'samples'), exist_ok=True)
    
    # Initialize models
    generator = Generator(
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        features=config.generator_features
    ).to(device)
    
    discriminator = Discriminator(
        input_channels=config.input_channels,
        image_channels=config.output_channels,
        features=config.discriminator_features
    ).to(device)
    
    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    # Optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.learning_rate_g,
        betas=(config.beta1, config.beta2)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate_d,
        betas=(config.beta1, config.beta2)
    )
    
    # Data loaders
    train_loader = get_dataloader(
        config.train_sketch_dir,
        config.train_image_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        mode='train'
    )
    
    val_loader = None
    if os.path.exists(config.val_sketch_dir):
        val_loader = get_dataloader(
            config.val_sketch_dir,
            config.val_image_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            shuffle=False,
            mode='val'
        )
    
    # TensorBoard writer
    writer = SummaryWriter(config.log_dir)
    
    # Training loop
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_gan_loss = 0
        epoch_l1_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (sketch, real_image) in enumerate(pbar):
            sketch = sketch.to(device)
            real_image = real_image.to(device)
            
            # Train discriminator
            d_loss = train_discriminator(
                discriminator, generator, sketch, real_image,
                optimizer_d, criterion_gan, device, config
            )
            
            # Train generator
            g_loss, gan_loss, l1_loss = train_generator(
                generator, discriminator, sketch, real_image,
                optimizer_g, criterion_gan, criterion_l1, device, config
            )
            
            # Update metrics
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            epoch_gan_loss += gan_loss
            epoch_l1_loss += l1_loss
            
            # Logging
            global_step += 1
            if global_step % config.log_interval == 0:
                writer.add_scalar('Loss/Generator', g_loss, global_step)
                writer.add_scalar('Loss/Discriminator', d_loss, global_step)
                writer.add_scalar('Loss/GAN', gan_loss, global_step)
                writer.add_scalar('Loss/L1', l1_loss, global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f'{g_loss:.4f}',
                'D_Loss': f'{d_loss:.4f}',
                'L1': f'{l1_loss:.4f}'
            })
        
        # Average losses
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Generator Loss: {avg_g_loss:.4f}")
        print(f"Discriminator Loss: {avg_d_loss:.4f}")
        
        # Save sample images
        if (epoch + 1) % config.save_interval == 0:
            generator.eval()
            sample_sketch, sample_image = next(iter(train_loader))
            save_sample_images(
                generator, sample_sketch, sample_image, epoch + 1,
                os.path.join(config.log_dir, 'samples'), device
            )
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_checkpoint(generator, optimizer_g, epoch + 1, avg_g_loss, checkpoint_path)
        
        # Save best model
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            best_path = os.path.join(config.checkpoint_dir, 'generator_best.pth')
            save_checkpoint(generator, optimizer_g, epoch + 1, avg_g_loss, best_path)
            print(f"Saved best model with loss: {best_loss:.4f}")
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sketch-to-image generator')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                       help='Directory containing train/val folders')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.data_dir = args.data_dir
    config.train_sketch_dir = os.path.join(args.data_dir, 'train', 'sketches')
    config.train_image_dir = os.path.join(args.data_dir, 'train', 'images')
    config.val_sketch_dir = os.path.join(args.data_dir, 'val', 'sketches')
    config.val_image_dir = os.path.join(args.data_dir, 'val', 'images')
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.image_size = args.image_size
    config.learning_rate_g = args.lr
    config.learning_rate_d = args.lr
    
    train(config)

