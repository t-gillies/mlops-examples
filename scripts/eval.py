import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.eval import AngleLoss, plot_predictions, plot_training_curves, validate
from src.model import T92AnglePredictor
from src.train import save_real_image_snapshot
from src.utils import load_cfg


def main(config_path: str, run_dir: str, test_dataset_path: str | None = None) -> None:
    cfg = load_cfg(config_path)

    run_dir = Path(run_dir)
    snapshots_dir = run_dir / "snapshots"
    best_model_path = run_dir / "models" / "best_model.pt"
    history_path = run_dir / "history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = AngleLoss()

    dataset_path = Path(test_dataset_path) if test_dataset_path else Path("data/processed/test_ds.pt")
    test_ds = torch.load(dataset_path, weights_only=False)

    num_workers = 0 if os.name == "nt" else 4
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = T92AnglePredictor(
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        use_zenith_input=cfg["use_zenith_input"],
    ).to(device)

    existing = {}
    history = {"train": [], "val": []}
    if history_path.exists():
        with open(history_path, "r") as f:
            existing = json.load(f)
        history = existing.get("history", history)


    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    checkpoint = torch.load(best_model_path, weights_only=False)
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
        updated_history = dict(existing)
        updated_history.update({
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            'history': history,
            'test_metrics': {
                'loss': test_metrics['loss'],
                'azimuth_mae': test_metrics['azimuth_mae'],
                'elevation_mae': test_metrics['elevation_mae']
            }
        })
        json.dump(updated_history, f, indent=2)
    print(f"  History: {history_path}")
    
    print(f"\n  Best Model: {best_model_path}")
    print(f"  All Snapshots: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-dataset", default=None)
    args = parser.parse_args()
    main(args.config, args.run_dir, args.test_dataset)
