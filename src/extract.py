import torch
from torch.utils.data import Dataset
import torch
from pathlib import Path
import csv
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

# ============================================
# DATASET WITH SMART MIRRORING
# ============================================
class T92Dataset(Dataset):
    def __init__(self, data_dir, transform=None, use_mirroring=True):
        self.image_paths = []
        self.azimuths = []
        self.elevations = []
        self.zeniths = []
        self.transform = transform
        self.use_mirroring = use_mirroring
        
        data_dir = Path(data_dir)
        
        elevation_folders = sorted([f for f in data_dir.iterdir() 
                                    if f.is_dir() and f.name.startswith('elev_')])

        # elevation_folders = elevation_folders[:1]
        
        print(f"Found {len(elevation_folders)} elevation folders")
        
        for folder in elevation_folders:
            csv_path = folder / 'labels.csv'
            if not csv_path.exists():
                print(f"  Warning: No labels.csv in {folder.name}")
                continue
            
            count = 0
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = folder / row['filename']
                    if img_path.exists():
                        self.image_paths.append(img_path)
                        self.azimuths.append(float(row['azimuth']))
                        self.elevations.append(float(row['elevation']))
                        self.zeniths.append(float(row['zenith']))
                        count += 1
            
            print(f"  {folder.name}: {count} images")
        
        self.azimuths = np.array(self.azimuths)
        self.elevations = np.array(self.elevations)
        self.zeniths = np.array(self.zeniths)
        
        self.base_length = len(self.image_paths)
        
        print(f"\nBase dataset: {self.base_length} images")
        if self.use_mirroring:
            print(f"With mirroring: {self.base_length * 2} effective images")
        print(f"  Azimuth: {self.azimuths.min():.1f} - {self.azimuths.max():.1f}")
        print(f"  Elevation: {self.elevations.min():.1f} - {self.elevations.max():.1f}")
        print(f"  Zenith: {self.zeniths.min():.1f} - {self.zeniths.max():.1f}")
    
    def __len__(self):
        if self.use_mirroring:
            return self.base_length * 2
        return self.base_length
    
    def __getitem__(self, idx):
        is_mirrored = False
        if self.use_mirroring and idx >= self.base_length:
            is_mirrored = True
            idx = idx - self.base_length
        
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        azimuth = self.azimuths[idx]
        elevation = self.elevations[idx]
        zenith = self.zeniths[idx]
        
        if is_mirrored:
            image = TF.hflip(image)
            azimuth = (360.0 - azimuth) % 360.0
        
        if self.transform:
            image = self.transform(image)
        
        az_rad = np.radians(azimuth)
        az_sin = np.sin(az_rad)
        az_cos = np.cos(az_rad)
        
        elev_norm = elevation / 90.0
        zenith_norm = zenith / 90.0
        
        target = torch.tensor([az_sin, az_cos, elev_norm], dtype=torch.float32)
        zenith_tensor = torch.tensor([zenith_norm], dtype=torch.float32)
        
        return image, zenith_tensor, target