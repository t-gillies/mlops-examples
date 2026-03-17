import csv
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.transform import get_val_transforms
from src.eval import compute_errors



# ============================================
# REAL IMAGE SNAPSHOT
# ============================================
def predict_on_real_images(model, device, cfg):
    """Run inference on all real test images"""
    model.eval()
    transform = get_val_transforms(cfg)
    
    real_dir = cfg['real_test_dir']
    if not real_dir.exists():
        return []
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(real_dir.glob(ext))
    
    if not image_paths:
        return []
    
    results = []
    
    for img_path in sorted(image_paths):
        image = Image.open(img_path).convert('RGB')
        original_image = image.copy()
        
        image = image.resize((224, 224), Image.LANCZOS)
        image_tensor = transform(image).unsqueeze(0).to(device)
        zenith = torch.tensor([[cfg['real_zenith_estimate'] / 90.0]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(image_tensor, zenith)
        
        azimuth = torch.rad2deg(torch.atan2(output[0, 0], output[0, 1])).item() % 360
        elevation = output[0, 2].item() * 90
        
        results.append({
            'filename': img_path.name,
            'path': img_path,
            'image': original_image,
            'azimuth': round(azimuth, 1),
            'elevation': round(elevation, 1)
        })
    
    return results


def save_real_image_snapshot(
    model,
    device,
    epoch,
    val_metrics,
    snapshots_dir,
    cfg,
    checkpoint_payload=None,
):
    """Save visualization of predictions on real images"""
    
    results = predict_on_real_images(model, device, cfg)
    
    if not results:
        print("  No real test images found - skipping snapshot")
        return
    
    # Create epoch folder
    if isinstance(epoch, int):
        epoch_name = f"epoch_{epoch:03d}"
    else:
        epoch_name = f"epoch_{epoch}"
    
    epoch_dir = snapshots_dir / epoch_name

    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint_path = epoch_dir / "model.pt"
    if checkpoint_payload is None:
        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics
        }

    torch.save(checkpoint_payload, checkpoint_path)
    
    # Create grid visualization
    num_images = len(results)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    for i, r in enumerate(results):
        img = r['image'].resize((224, 224), Image.LANCZOS)
        axes[i].imshow(img)
        axes[i].set_title(f"{r['filename']}\nAz: {r['azimuth']:.1f}  El: {r['elevation']:.1f}", fontsize=10)
        axes[i].axis('off')
    
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    # Add epoch info
    fig.suptitle(
        f"Epoch {epoch} | Val Az: {val_metrics['azimuth_mae']:.2f} | Val El: {val_metrics['elevation_mae']:.2f}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    grid_path = epoch_dir / "predictions_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual images with predictions
    for r in results:
        img = r['image'].resize((224, 224), Image.LANCZOS)
        
        # Add text overlay
        draw = ImageDraw.Draw(img)
        text = f"Az: {r['azimuth']:.1f}  El: {r['elevation']:.1f}"
        
        # Draw text with background
        bbox = draw.textbbox((5, 5), text)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='black')
        draw.text((5, 5), text, fill='white')
        
        img.save(epoch_dir / f"pred_{r['filename']}")
    
    # Save predictions to CSV
    csv_path = epoch_dir / "predictions.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'azimuth', 'elevation', 'zenith_used'])
        for r in results:
            writer.writerow([r['filename'], r['azimuth'], r['elevation'], cfg['real_zenith_estimate']])
    
    print(f"  Snapshot saved to {epoch_dir}")





# ============================================
# TRAINING FUNCTIONS
# ============================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_az_err = 0
    total_elev_err = 0
    count = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, zeniths, targets in pbar:
        images = images.to(device)
        zeniths = zeniths.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, zeniths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        errors = compute_errors(outputs, targets)
        
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_az_err += errors['azimuth_mae'] * bs
        total_elev_err += errors['elevation_mae'] * bs
        count += bs
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'az': f'{errors["azimuth_mae"]:.1f}',
            'el': f'{errors["elevation_mae"]:.1f}'
        })
    
    return {
        'loss': total_loss / count,
        'azimuth_mae': total_az_err / count,
        'elevation_mae': total_elev_err / count
    }
