"""
Configuration file for the sketch-to-image generation project
"""
import torch

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    data_dir = './dataset'
    train_sketch_dir = './dataset/train/sketches'
    train_image_dir = './dataset/train/images'
    val_sketch_dir = './dataset/val/sketches'
    val_image_dir = './dataset/val/images'
    
    # Model parameters
    input_channels = 1  # Grayscale sketch
    output_channels = 3  # RGB image
    generator_features = 64
    discriminator_features = 64
    
    # Training parameters
    batch_size = 4
    num_epochs = 100
    learning_rate_g = 0.0002
    learning_rate_d = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    
    # Loss weights
    lambda_l1 = 100.0  # L1 loss weight
    lambda_gan = 1.0   # GAN loss weight
    
    # Image parameters
    image_size = 256
    
    # Checkpoint and logging
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    save_interval = 10
    log_interval = 100
    
    # Inference
    checkpoint_path = './checkpoints/generator_best.pth'

