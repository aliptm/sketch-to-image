"""
Dataset class for loading sketch-image pairs
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SketchImageDataset(Dataset):
    """Dataset for sketch-image pairs"""
    
    def __init__(self, sketch_dir, image_dir, image_size=256, mode='train'):
        self.sketch_dir = sketch_dir
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Get list of files
        self.sketch_files = sorted([f for f in os.listdir(sketch_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure same number of files
        assert len(self.sketch_files) == len(self.image_files), \
            f"Number of sketches ({len(self.sketch_files)}) and images ({len(self.image_files)}) must match"
        
        # Transformations
        if mode == 'train':
            self.transform_sketch = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
            
            self.transform_image = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        else:
            # Validation/test: no augmentation
            self.transform_sketch = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            self.transform_image = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.sketch_files)
    
    def __getitem__(self, idx):
        # Load sketch
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx])
        sketch = Image.open(sketch_path).convert('RGB')
        sketch = self.transform_sketch(sketch)
        
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform_image(image)
        
        return sketch, image


def get_dataloader(sketch_dir, image_dir, batch_size=4, image_size=256, 
                   shuffle=True, num_workers=2, mode='train'):
    """Create DataLoader for sketch-image pairs"""
    dataset = SketchImageDataset(sketch_dir, image_dir, image_size, mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

