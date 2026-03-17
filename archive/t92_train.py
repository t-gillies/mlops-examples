""" T92 Angle Prediction - Training with Real Image Monitoring
    
    Features:
    - Smart horizontal mirroring (doubles dataset)
    - Blur + noise augmentation for sim-to-real
    - Zenith-conditioned prediction
    - Sin/cos azimuth encoding
    - Saves snapshots on real images every epoch
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch

from src.utils import load_cfg, seed_all
from src.extract import T92Dataset
from src.transform import TransformSubset, get_train_transforms, get_val_transforms
from src.model import T92AnglePredictor
from src.train import train_epoch, save_real_image_snapshot
from src.eval import AngleLoss, plot_predictions, plot_training_curves, validate




# ============================================
# MAIN TRAINING
# ============================================
def train(cfg):

    print("=" * 60)
    print("T92 ANGLE PREDICTION TRAINING")
    print("With Real Image Monitoring")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # #Set common seeds
    seed_all(cfg['seed'])

    # Create run directory for snapshots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg["snapshots_dir"] / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    snapshots_dir = run_dir / "snapshots"
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSnapshots will be saved to: {run_dir}")
    
    # Check for real test images
    if cfg['real_test_dir'].exists():
        real_images = list(cfg['real_test_dir'].glob('*.png')) + \
                      list(cfg['real_test_dir'].glob('*.jpg')) + \
                      list(cfg['real_test_dir'].glob('*.jpeg'))
        print(f"Found {len(real_images)} real test images")
    else:
        print(f"WARNING: Real test directory not found: {cfg['real_test_dir']}")
        print("Create it and add test images to enable monitoring")

    # ============================================
    # DATA EXTRACTION
    # ============================================
    # Reads raw image paths / labels from the configured data directory
    # into the base dataset object.
    print("\nLoading dataset...")
    full_dataset = T92Dataset(
        cfg['data_dir'], 
        transform=None, 
        use_mirroring=False
    )
    
    if len(full_dataset) == 0:
        print("ERROR: No data found!")
        return None
    

    # ============================================
    # DATA TRANSFORMATION
    # ============================================
    # Applies train/validation transforms and wraps splits with the
    # subset dataset used for mirroring/target shaping.

    
    total = full_dataset.base_length
    indices = list(range(total))
    
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(cfg['val_split'] + cfg['test_split']),
        random_state=cfg['seed']
    )
    
    val_ratio = cfg['val_split'] / (cfg['val_split'] + cfg['test_split'])
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio),
        random_state=cfg['seed']
    )

    print(f"\nBase split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")


    train_ds = TransformSubset(
        full_dataset, train_indices, 
        get_train_transforms(cfg), 
        use_mirroring=cfg['use_mirroring']
    )
    
    val_ds = TransformSubset(
        full_dataset, val_indices,
        get_val_transforms(cfg),
        use_mirroring=False
    )
    
    test_ds = TransformSubset(
        full_dataset, test_indices,
        get_val_transforms(cfg),
        use_mirroring=False
    )

    print(f"Effective sizes: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")


    # ============================================
    # MODEL TRAINING
    # ============================================
    # Builds the model, optimizer, and scheduler, then runs
    # the optimization loop.

    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"\nCreating model: {cfg['backbone']}")
    model = T92AnglePredictor(
        backbone=cfg['backbone'],
        pretrained=cfg['pretrained'],
        use_zenith_input=cfg['use_zenith_input']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    criterion = AngleLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg['learning_rate'], 
        weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}

    best_model_path = model_dir / "best_model.pt"
    # Save cfg
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}, f, indent=2)
    
    print(f"\nTraining for up to {cfg['epochs']} epochs...")
    print(f"Mirroring: {cfg['use_mirroring']}")
    print(f"Real image snapshots every {cfg['snapshot_every_n_epochs']} epoch(s)")
    print("=" * 60)
    

    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        print("-" * 40)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        print(f"Train | Loss: {train_metrics['loss']:.4f} | Az: {train_metrics['azimuth_mae']:.2f} | El: {train_metrics['elevation_mae']:.2f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f} | Az: {val_metrics['azimuth_mae']:.2f} | El: {val_metrics['elevation_mae']:.2f}")
        
        history['train'].append({
            'loss': train_metrics['loss'],
            'azimuth_mae': train_metrics['azimuth_mae'],
            'elevation_mae': train_metrics['elevation_mae']
        })
        history['val'].append({
            'loss': val_metrics['loss'],
            'azimuth_mae': val_metrics['azimuth_mae'],
            'elevation_mae': val_metrics['elevation_mae']
        })
        
        # Save real image snapshot
        if (epoch + 1) % cfg['snapshot_every_n_epochs'] == 0:
            save_real_image_snapshot(model, device, epoch + 1, {
                'loss': val_metrics['loss'],
                'azimuth_mae': val_metrics['azimuth_mae'],
                'elevation_mae': val_metrics['elevation_mae']
            }, snapshots_dir, cfg)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': {
                    'loss': val_metrics['loss'],
                    'azimuth_mae': val_metrics['azimuth_mae'],
                    'elevation_mae': val_metrics['elevation_mae']
                },
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            }, best_model_path)
            
            print("Saved best model")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{cfg['patience']})")
            
            if patience_counter >= cfg['patience']:
                print("\nEarly stopping!")
                break
    
    # ============================================
    # EVALUATION
    # ============================================
    # test evaluation runs after the best checkpoint is reloaded.

    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss:          {test_metrics['loss']:.4f}")
    print(f"  Azimuth MAE:   {test_metrics['azimuth_mae']:.2f}")
    print(f"  Elevation MAE: {test_metrics['elevation_mae']:.2f}")
    
    # Final snapshot
    save_real_image_snapshot(model, device, 'final', {
        'loss': test_metrics['loss'],
        'azimuth_mae': test_metrics['azimuth_mae'],
        'elevation_mae': test_metrics['elevation_mae']
    }, snapshots_dir, cfg)
    
    curves_path = run_dir / "training_curves.png"
    plot_training_curves(history, curves_path)
    print(f"\n  Curves: {curves_path}")
    
    predictions_path = run_dir / "test_predictions.png"
    plot_predictions(test_metrics['predictions'], predictions_path)
    print(f"  Predictions: {predictions_path}")
    
    history_path = run_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            'history': history,
            'test_metrics': {
                'loss': test_metrics['loss'],
                'azimuth_mae': test_metrics['azimuth_mae'],
                'elevation_mae': test_metrics['elevation_mae']
            }
        }, f, indent=2)
    print(f"  History: {history_path}")
    
    print(f"\n  Best Model: {best_model_path}")
    print(f"  All Snapshots: {run_dir}")



    # ============================================
    # LOGGING
    # ============================================
    # Log with MLFlow


    mlflow.set_tracking_uri(os.path.expandvars(cfg["tracking_uri"]))
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_azimuth_mae": test_metrics["azimuth_mae"],
            "test_elevation_mae": test_metrics["elevation_mae"],
        })
        mlflow.log_artifacts(str(run_dir), artifact_path="eval")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="t92_model",
            registered_model_name=cfg['registered_model_name'],
        )
    
    return model


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    model = train(cfg)
