
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================
# LOSS AND METRICS
# ============================================
class AngleLoss(nn.Module):
    def __init__(self, azimuth_weight=1.0, elevation_weight=1.0):
        super().__init__()
        self.azimuth_weight = azimuth_weight
        self.elevation_weight = elevation_weight
    
    def forward(self, pred, target):
        az_loss = nn.functional.mse_loss(pred[:, :2], target[:, :2])
        elev_loss = nn.functional.mse_loss(pred[:, 2], target[:, 2])
        return self.azimuth_weight * az_loss + self.elevation_weight * elev_loss


def compute_errors(pred, target):
    with torch.no_grad():
        pred_az = torch.rad2deg(torch.atan2(pred[:, 0], pred[:, 1])) % 360
        target_az = torch.rad2deg(torch.atan2(target[:, 0], target[:, 1])) % 360
        
        az_error = torch.abs(pred_az - target_az)
        az_error = torch.min(az_error, 360 - az_error)
        
        pred_elev = pred[:, 2] * 90
        target_elev = target[:, 2] * 90
        elev_error = torch.abs(pred_elev - target_elev)
        
        return {
            'azimuth_mae': az_error.mean().item(),
            'elevation_mae': elev_error.mean().item(),
            'pred_az': pred_az,
            'pred_elev': pred_elev,
            'target_az': target_az,
            'target_elev': target_elev
        }


# ============================================
# PLOTTING
# ============================================
def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train']) + 1)
    
    axes[0].plot(epochs, [h['loss'] for h in history['train']], label='Train')
    axes[0].plot(epochs, [h['loss'] for h in history['val']], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, [h['azimuth_mae'] for h in history['train']], label='Train')
    axes[1].plot(epochs, [h['azimuth_mae'] for h in history['val']], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (degrees)')
    axes[1].set_title('Azimuth Error')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(epochs, [h['elevation_mae'] for h in history['train']], label='Train')
    axes[2].plot(epochs, [h['elevation_mae'] for h in history['val']], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE (degrees)')
    axes[2].set_title('Elevation Error')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(predictions, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(predictions['target_az'], predictions['pred_az'], alpha=0.1, s=1)
    axes[0].plot([0, 360], [0, 360], 'r--', linewidth=2)
    axes[0].set_xlabel('True Azimuth')
    axes[0].set_ylabel('Predicted Azimuth')
    axes[0].set_title('Azimuth Predictions')
    axes[0].set_xlim(0, 360)
    axes[0].set_ylim(0, 360)
    axes[0].grid(True)
    
    axes[1].scatter(predictions['target_elev'], predictions['pred_elev'], alpha=0.1, s=1)
    axes[1].plot([0, 90], [0, 90], 'r--', linewidth=2)
    axes[1].set_xlabel('True Elevation')
    axes[1].set_ylabel('Predicted Elevation')
    axes[1].set_title('Elevation Predictions')
    axes[1].set_xlim(0, 90)
    axes[1].set_ylim(0, 90)
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_az_err = 0
    total_elev_err = 0
    count = 0
    
    all_pred_az = []
    all_pred_elev = []
    all_target_az = []
    all_target_elev = []
    
    with torch.no_grad():
        for images, zeniths, targets in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            zeniths = zeniths.to(device)
            targets = targets.to(device)
            
            outputs = model(images, zeniths)
            loss = criterion(outputs, targets)
            
            errors = compute_errors(outputs, targets)
            
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_az_err += errors['azimuth_mae'] * bs
            total_elev_err += errors['elevation_mae'] * bs
            count += bs
            
            all_pred_az.extend(errors['pred_az'].cpu().tolist())
            all_pred_elev.extend(errors['pred_elev'].cpu().tolist())
            all_target_az.extend(errors['target_az'].cpu().tolist())
            all_target_elev.extend(errors['target_elev'].cpu().tolist())
    
    return {
        'loss': total_loss / count,
        'azimuth_mae': total_az_err / count,
        'elevation_mae': total_elev_err / count,
        'predictions': {
            'pred_az': all_pred_az,
            'pred_elev': all_pred_elev,
            'target_az': all_target_az,
            'target_elev': all_target_elev
        }
    }
