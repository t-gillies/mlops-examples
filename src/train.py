import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    for key in ("CI_COMMIT_SHA", "GIT_COMMIT", "GITHUB_SHA"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def plot_confusion(cm, out_path: Path) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(y_true, y_score, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_curve(y_true, y_score, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, out_path: Path, top_n: int = 15) -> None:
    order = np.argsort(importances)[::-1]
    top_idx = order[:top_n]
    fig = plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
    )
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg["mlflow"]["experiment_name"])
    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME", cfg["mlflow"]["registered_model_name"]
    )
    model_stage = os.environ.get("MLFLOW_MODEL_STAGE", cfg["mlflow"].get("model_stage", "")).strip()

    data_path = Path(cfg["data"]["path"])
    out_dir = Path(cfg["artifacts"]["out_dir"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Did you run `dvc pull` (or generate + dvc add)?"
        )

    data_hash = sha256_file(data_path)
    git_sha = get_git_sha()

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=float(cfg["train"]["test_size"]),
        random_state=int(cfg["train"]["seed"]),
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=int(cfg["train"]["n_estimators"]),
        max_depth=(
            None
            if str(cfg["train"]["max_depth"]).lower() in ("none", "null")
            else int(cfg["train"]["max_depth"])
        ),
        min_samples_split=int(cfg["train"]["min_samples_split"]),
        min_samples_leaf=int(cfg["train"]["min_samples_leaf"]),
        max_features=cfg["train"]["max_features"],
        random_state=int(cfg["train"]["seed"]),
        n_jobs=None,
    )
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="macro")
    precision = precision_score(yte, preds)
    recall = recall_score(yte, preds)
    roc_auc = roc_auc_score(yte, probs)
    pr_auc = average_precision_score(yte, probs)
    cm = confusion_matrix(yte, preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    cm_path = out_dir / "confusion_matrix.png"
    roc_path = out_dir / "roc_curve.png"
    pr_path = out_dir / "pr_curve.png"
    fi_path = out_dir / "feature_importance.png"

    metrics = {
        "val_accuracy": acc,
        "val_f1_macro": f1,
        "val_precision": precision,
        "val_recall": recall,
        "val_roc_auc": roc_auc,
        "val_pr_auc": pr_auc,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, cm_path)
    plot_roc_curve(yte, probs, roc_path)
    plot_pr_curve(yte, probs, pr_path)
    plot_feature_importance(X.columns.to_list(), model.feature_importances_, fi_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tag("git_sha", git_sha)
        mlflow.set_tag("data_sha256", data_hash)
        mlflow.set_tag("pipeline", "mlops-examples")
        mlflow.set_tag("dataset_path", str(data_path))

        mlflow.log_param("model_type", cfg["train"]["model_type"])
        mlflow.log_param("seed", cfg["train"]["seed"])
        mlflow.log_param("test_size", cfg["train"]["test_size"])
        mlflow.log_param("n_estimators", cfg["train"]["n_estimators"])
        mlflow.log_param("max_depth", cfg["train"]["max_depth"])
        mlflow.log_param("min_samples_split", cfg["train"]["min_samples_split"])
        mlflow.log_param("min_samples_leaf", cfg["train"]["min_samples_leaf"])
        mlflow.log_param("max_features", cfg["train"]["max_features"])

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="eval")
        mlflow.log_artifact(str(cm_path), artifact_path="eval")
        mlflow.log_artifact(str(roc_path), artifact_path="eval")
        mlflow.log_artifact(str(pr_path), artifact_path="eval")
        mlflow.log_artifact(str(fi_path), artifact_path="eval")

        input_example = Xtr.head(5)
        signature = mlflow.models.infer_signature(Xtr, model.predict(Xtr))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        model_version = None
        if model_info.model_uri and registered_model_name:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            run_versions = [v for v in versions if getattr(v, "run_id", None) == run_id]
            if run_versions:
                run_versions.sort(key=lambda v: int(v.version))
                model_version = run_versions[-1].version

            if model_stage and model_version is not None:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=model_version,
                    stage=model_stage,
                )

        print("MLflow run complete")
        print(f"Run ID: {run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Registered model: {registered_model_name}")
        if model_version is not None:
            print(f"Model version: {model_version}")
        if model_stage and model_version is not None:
            print(f"Stage '{model_stage}' -> v{model_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
