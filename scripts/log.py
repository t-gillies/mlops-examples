import argparse
import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from src.utils import get_git_sha, sha256_file


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    split_dir = Path(cfg["data"]["split_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    metrics_dir = Path(cfg["artifacts"]["metrics_dir"])

    tracking_uri = os.path.expandvars(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg["mlflow"]["experiment_name"])
    registered_model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", cfg["mlflow"]["registered_model_name"])

    train_df = pd.read_csv(split_dir / "train.csv")
    metrics = json.loads((metrics_dir / "metrics.json").read_text())

    with (model_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)

    X_train = train_df.drop(columns=["target", "event_timestamp", "patient_id"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        tags = {
            **cfg["mlflow"].get("tags", {}),
            "git_sha": get_git_sha(),
            "raw_data_sha256": sha256_file(Path(cfg["data"]["raw_path"])),
            "processed_data_sha256": sha256_file(Path(cfg["data"]["processed_path"])),
        }
        for key, value in tags.items():
            if value is not None:
                mlflow.set_tag(key, value)

        if "train" in cfg:
            mlflow.log_params(cfg["train"])
        if "split" in cfg:
            mlflow.log_params({f"split_{k}": v for k, v in cfg["split"].items()})

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(metrics_dir), artifact_path="metrics")

        input_example = X_train.head(5)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
