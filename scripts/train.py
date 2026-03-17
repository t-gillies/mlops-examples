import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.eval import AngleLoss, validate
from src.model import T92AnglePredictor
from src.train import save_real_image_snapshot, train_epoch
from src.utils import load_cfg, seed_all


def _serialize_cfg(cfg: dict) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}


def _infer_run_dir(resume_path: Path) -> Path:
    if resume_path.parent.name == "models":
        return resume_path.parent.parent

    if resume_path.parent.parent.name == "snapshots":
        return resume_path.parent.parent.parent

    raise ValueError(
        f"Could not infer run directory from checkpoint path: {resume_path}"
    )


def _completed_epochs_from_checkpoint(checkpoint: dict, checkpoint_path: Path) -> int:
    if "completed_epochs" in checkpoint:
        return int(checkpoint["completed_epochs"])

    if "epoch_index" in checkpoint:
        return int(checkpoint["epoch_index"]) + 1

    epoch = checkpoint.get("epoch")
    if epoch is None:
        return 0

    if checkpoint_path.name == "best_model.pt" or checkpoint_path.parent.name == "models":
        return int(epoch) + 1

    return int(epoch)


def _write_history(
    history_path: Path,
    cfg: dict,
    history: dict,
    input_dir: Path,
    best_model_path: Path,
) -> None:
    payload = {
        "config": _serialize_cfg(cfg),
        "history": history,
        "test_dataset_path": str(input_dir / "test_ds.pt"),
        "best_model_path": str(best_model_path),
    }

    with open(history_path, "w") as f:
        json.dump(payload, f, indent=2)


def main(
    config_path: str,
    in_dir: str | None = None,
    resume_from: str | None = None,
) -> Path:
    cfg = load_cfg(config_path)
    input_dir = Path(in_dir) if in_dir else Path("data/processed")

    print("=" * 60)
    print("T92 TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    seed_all(cfg["seed"])

    resume_path = Path(resume_from) if resume_from else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    if resume_path is not None:
        run_dir = _infer_run_dir(resume_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = cfg["snapshots_dir"] / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    snapshots_dir = run_dir / "snapshots"
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    if resume_path is not None:
        print(f"\nResuming from: {resume_path}")
        print(f"Continuing run in: {run_dir}")
    else:
        print(f"\nSnapshots will be saved to: {run_dir}")

    if cfg["real_test_dir"].exists():
        real_images = list(cfg["real_test_dir"].glob("*.png")) + \
            list(cfg["real_test_dir"].glob("*.jpg")) + \
            list(cfg["real_test_dir"].glob("*.jpeg"))
        print(f"Found {len(real_images)} real test images")
    else:
        print(f"WARNING: Real test directory not found: {cfg['real_test_dir']}")
        print("Create it and add test images to enable monitoring")

    train_ds = torch.load(input_dir / "train_ds.pt", weights_only=False)
    val_ds = torch.load(input_dir / "val_ds.pt", weights_only=False)
    num_workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"\nCreating model: {cfg['backbone']}")
    model = T92AnglePredictor(
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        use_zenith_input=cfg["use_zenith_input"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = AngleLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}
    start_epoch = 0

    best_model_path = model_dir / "best_model.pt"
    history_path = run_dir / "history.json"
    config_out_path = run_dir / "config.json"
    with open(config_out_path, "w") as f:
        json.dump(_serialize_cfg(cfg), f, indent=2)

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f).get("history", history)

    best_completed_epochs = 0
    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        best_val_loss = best_checkpoint.get(
            "best_val_loss",
            best_checkpoint.get("val_metrics", {}).get("loss", float("inf")),
        )
        best_completed_epochs = _completed_epochs_from_checkpoint(
            best_checkpoint, best_model_path
        )

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("Resume checkpoint does not include optimizer state; starting optimizer fresh")

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print("Resume checkpoint does not include scheduler state; starting scheduler fresh")

        start_epoch = _completed_epochs_from_checkpoint(checkpoint, resume_path)
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        patience_counter = checkpoint.get(
            "patience_counter",
            max(0, start_epoch - best_completed_epochs),
        )
        history = checkpoint.get("history", history)

        print(f"Restarting at epoch {start_epoch + 1}/{cfg['epochs']}")

    print(f"\nTraining for up to {cfg['epochs']} epochs...")
    print(f"Mirroring: {cfg['use_mirroring']}")
    print(f"Real image snapshots every {cfg['snapshot_every_n_epochs']} epoch(s)")
    print("=" * 60)

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        print("-" * 40)

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        print(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Az: {train_metrics['azimuth_mae']:.2f} | "
            f"El: {train_metrics['elevation_mae']:.2f}"
        )
        print(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Az: {val_metrics['azimuth_mae']:.2f} | "
            f"El: {val_metrics['elevation_mae']:.2f}"
        )

        history["train"].append(
            {
                "loss": train_metrics["loss"],
                "azimuth_mae": train_metrics["azimuth_mae"],
                "elevation_mae": train_metrics["elevation_mae"],
            }
        )
        history["val"].append(
            {
                "loss": val_metrics["loss"],
                "azimuth_mae": val_metrics["azimuth_mae"],
                "elevation_mae": val_metrics["elevation_mae"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "epoch_index": epoch,
                    "completed_epochs": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "history": history,
                    "val_metrics": {
                        "loss": val_metrics["loss"],
                        "azimuth_mae": val_metrics["azimuth_mae"],
                        "elevation_mae": val_metrics["elevation_mae"],
                    },
                    "config": _serialize_cfg(cfg),
                },
                best_model_path,
            )

            print("Saved best model")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{cfg['patience']})")
        checkpoint_payload = {
            "epoch": epoch + 1,
            "epoch_index": epoch,
            "completed_epochs": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "history": history,
            "val_metrics": {
                "loss": val_metrics["loss"],
                "azimuth_mae": val_metrics["azimuth_mae"],
                "elevation_mae": val_metrics["elevation_mae"],
            },
            "config": _serialize_cfg(cfg),
        }

        if (epoch + 1) % cfg["snapshot_every_n_epochs"] == 0:
            save_real_image_snapshot(
                model,
                device,
                epoch + 1,
                checkpoint_payload["val_metrics"],
                snapshots_dir,
                cfg,
                checkpoint_payload=checkpoint_payload,
            )

        _write_history(history_path, cfg, history, input_dir, best_model_path)

        if patience_counter >= cfg["patience"]:
            print("\nEarly stopping!")
            break

    _write_history(history_path, cfg, history, input_dir, best_model_path)


    print(f"\nBest model: {best_model_path}")
    print(f"History: {history_path}")


    


    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    main(args.config, args.in_dir, args.resume_from)
