import torch
import torch.nn as nn
from torchvision import models


# ============================================
# MODEL
# ============================================
class T92AnglePredictor(nn.Module):
    def __init__(self, backbone='densenet121', pretrained=True, use_zenith_input=True):
        super().__init__()
        
        self.use_zenith_input = use_zenith_input
        
        if backbone == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        if use_zenith_input:
            num_features += 1
        
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    
    def forward(self, image, zenith=None):
        features = self.backbone(image)
        
        if self.use_zenith_input and zenith is not None:
            features = torch.cat([features, zenith], dim=1)
        
        return self.head(features)