
import csv
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np
from PIL import Image


# ============================================
# CUSTOM AUGMENTATIONS
# ============================================
class RandomBlur:
    def __init__(self, p=0.5, kernel_range=(3, 7)):
        self.p = p
        self.kernel_range = kernel_range
    
    def __call__(self, img):
        if random.random() < self.p:
            k = random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
            return TF.gaussian_blur(img, kernel_size=k)
        return img


class RandomNoise:
    def __init__(self, p=0.3, std_range=(0.01, 0.05)):
        self.p = p
        self.std_range = std_range
    
    def __call__(self, tensor):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std
            return torch.clamp(tensor + noise, 0, 1)
        return tensor



# ============================================
# DATA TRANSFORMS
# ============================================
def get_train_transforms(cfg):
    return transforms.Compose([
        transforms.Resize(cfg['image_size']),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=cfg['grayscale_prob']),
        RandomBlur(p=cfg['blur_prob'], kernel_range=cfg['blur_kernel_range']),
        transforms.ToTensor(),
        RandomNoise(p=cfg['noise_prob'], std_range=cfg['noise_std_range']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(cfg):
    return transforms.Compose([
        transforms.Resize(cfg['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================
# SUBSET WRAPPER
# ============================================
class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform, use_mirroring=True):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.use_mirroring = use_mirroring
        self.base_length = dataset.base_length
    
    def __len__(self):
        if self.use_mirroring:
            return len(self.indices) * 2
        return len(self.indices)
    
    def __getitem__(self, idx):
        is_mirrored = False
        if self.use_mirroring and idx >= len(self.indices):
            is_mirrored = True
            idx = idx - len(self.indices)
        
        real_idx = self.indices[idx]
        
        image = Image.open(self.dataset.image_paths[real_idx]).convert('RGB')
        
        azimuth = self.dataset.azimuths[real_idx]
        elevation = self.dataset.elevations[real_idx]
        zenith = self.dataset.zeniths[real_idx]
        
        if is_mirrored:
            image = TF.hflip(image)
            azimuth = (360.0 - azimuth) % 360.0
        
        if self.transform:
            image = self.transform(image)
        
        az_rad = np.radians(azimuth)
        target = torch.tensor([
            np.sin(az_rad),
            np.cos(az_rad),
            elevation / 90.0
        ], dtype=torch.float32)
        
        zenith_tensor = torch.tensor([zenith / 90.0], dtype=torch.float32)
        
        return image, zenith_tensor, target