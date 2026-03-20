from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlops_examples.config import load_config
from mlops_examples.utils import get_git_sha, read_dvc_hash, sha256_file


def log_run(config_path: str) -> None:
    cfg = load_config(config_path)

    split_dir = Path(cfg["data"]["split_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    metrics_dir = Path(cfg["artifacts"]["metrics_dir"])
    snapshot_dir = Path(cfg["features"]["snapshot_dir"])
    manifest_path = Path(cfg["features"]["manifest_path"])
    config_artifact_path = Path(config_path)
    uv_lock_path = Path("uv.lock")
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    snapshot_dvc_path = snapshot_dir.with_suffix(".dvc")

    tracking_uri = os.path.expandvars(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg["mlflow"]["experiment_name"])
    registered_model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", cfg["mlflow"]["registered_model_name"])

    train_df = pd.read_csv(split_dir / "train.csv")
    metrics = json.loads((metrics_dir / "metrics.json").read_text())

    with (model_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    x_train = train_df.drop(columns=["target", "event_timestamp", "patient_id"])

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    feature_snapshot_id = manifest.get("feature_snapshot_id")
    feature_snapshot_dvc_hash = read_dvc_hash(snapshot_dvc_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        tags = {
            **cfg["mlflow"].get("tags", {}),
            "git_sha": get_git_sha(),
            "raw_data_sha256": sha256_file(Path(cfg["data"]["raw_path"])),
            "processed_data_sha256": sha256_file(Path(cfg["data"]["processed_path"])),
            "feature_service_name": cfg["features"]["feature_service_name"],
            "feature_snapshot_id": feature_snapshot_id,
            "feature_snapshot_manifest_sha256": (
                sha256_file(manifest_path) if manifest_path.exists() else None
            ),
            "feature_snapshot_dvc_hash": feature_snapshot_dvc_hash,
            "train_dataset_sha256": sha256_file(train_path),
            "test_dataset_sha256": sha256_file(test_path),
        }
        for key, value in tags.items():
            if value is not None:
                mlflow.set_tag(key, value)

        if "train" in cfg:
            mlflow.log_params(cfg["train"])
        if "split" in cfg:
            mlflow.log_params({f"split_{key}": value for key, value in cfg["split"].items()})

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(metrics_dir), artifact_path="metrics")
        if config_artifact_path.exists():
            mlflow.log_artifact(str(config_artifact_path), artifact_path="config")
        if manifest_path.exists():
            mlflow.log_artifact(str(manifest_path), artifact_path="lineage")
        if uv_lock_path.exists():
            mlflow.log_artifact(str(uv_lock_path), artifact_path="environment")

        input_example = x_train.head(5)
        signature = mlflow.models.infer_signature(x_train, model.predict(x_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        print("MLflow run complete")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Registered model: {registered_model_name}")
