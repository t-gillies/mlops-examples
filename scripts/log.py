import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from src.model import T92AnglePredictor
from src.utils import get_git_sha, load_cfg, sha256_file



def main(config_path: str, run_dir: str) -> None:
    cfg = load_cfg(config_path)
    run_dir = Path(run_dir)

    history_path = run_dir / "history.json"
    best_model_path = run_dir / "models" / "best_model.pt"

    history = {}
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)

    test_metrics = history.get("test_metrics", {})
    training_history = history.get("history", {"train": [], "val": []})
    test_dataset_path = Path(history.get("test_dataset_path", "data/processed/test_ds.pt"))

    tags = {
        "git_sha": get_git_sha(),
        "dataset_path": str(test_dataset_path),
    }
    if test_dataset_path.exists():
        tags["data_sha256"] = sha256_file(test_dataset_path)
    if best_model_path.exists():
        tags["model_sha256"] = sha256_file(best_model_path)


    model = T92AnglePredictor(
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        use_zenith_input=cfg["use_zenith_input"],
    )
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()

    mlflow.set_tracking_uri(os.path.expandvars(cfg["tracking_uri"]))
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in cfg.items()})

        for key, value in tags.items():
            mlflow.set_tag(key, value)

        for step, metrics in enumerate(training_history.get("train", []), start=1):
            for key, value in metrics.items():
                mlflow.log_metric(f"train_{key}", value, step=step)

        for step, metrics in enumerate(training_history.get("val", []), start=1):
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=step)

        if test_metrics:
            mlflow.log_metrics({
                "test_loss": test_metrics["loss"],
                "test_azimuth_mae": test_metrics["azimuth_mae"],
                "test_elevation_mae": test_metrics["elevation_mae"],
            })

        mlflow.log_artifacts(str(run_dir), artifact_path="eval")
        mlflow.pytorch.log_model(
            pytorch_model= model,
            artifact_path="t92_model",
            registered_model_name=cfg["registered_model_name"],
        )

    print("Logged run to MLflow")
    print(f"  Run dir: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    main(args.config, args.run_dir)
