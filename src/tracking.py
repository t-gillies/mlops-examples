"""Config-driven MLflow logging helper.

Provides :func:`log_run` which handles all MLflow interactions in a single
call: setting tags, logging params/metrics/artifacts, registering the model,
and optionally promoting it to a stage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator


def _resolve_mlflow_config(cfg: dict[str, Any]) -> dict[str, str]:
    """Resolve MLflow connection settings with env-var overrides.

    Environment variables take precedence over config values:
    - ``MLFLOW_TRACKING_URI``
    - ``MLFLOW_EXPERIMENT_NAME``
    - ``MLFLOW_REGISTERED_MODEL_NAME``
    - ``MLFLOW_MODEL_STAGE``
    """
    mlflow_cfg = cfg["mlflow"]
    return {
        "tracking_uri": os.environ.get(
            "MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"]
        ),
        "experiment_name": os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", mlflow_cfg["experiment_name"]
        ),
        "registered_model_name": os.environ.get(
            "MLFLOW_REGISTERED_MODEL_NAME", mlflow_cfg["registered_model_name"]
        ),
        "model_stage": os.environ.get(
            "MLFLOW_MODEL_STAGE", mlflow_cfg.get("model_stage", "")
        ).strip(),
    }


def log_run(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, float],
    artifact_dir: Path,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    tags: dict[str, str],
) -> None:
    """Log a complete training run to MLflow.

    This single function replaces the manual sequence of ``set_tag``,
    ``log_param``, ``log_metrics``, ``log_artifact``, ``log_model``, and
    model-version promotion calls.

    Parameters
    ----------
    cfg : dict
        The full parsed YAML config.  Keys used:

        - ``cfg["mlflow"]`` — connection and registry settings
        - ``cfg["train"]``  — all keys are logged as params automatically
    metrics : dict
        Metric name → value pairs to log.
    artifact_dir : Path
        Directory whose contents are logged under the ``eval/`` artifact path.
    model : BaseEstimator
        The trained sklearn model to log and register.
    X_train : DataFrame
        Training features used to infer the model signature and input example.
    tags : dict
        Runtime tags (e.g. ``git_sha``, ``data_sha256``) merged with any
        static tags declared in ``cfg["mlflow"]["tags"]``.
    """
    resolved = _resolve_mlflow_config(cfg)

    mlflow.set_tracking_uri(resolved["tracking_uri"])
    mlflow.set_experiment(resolved["experiment_name"])

    registered_model_name = resolved["registered_model_name"]
    model_stage = resolved["model_stage"]

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ── Tags: merge config-level + runtime ──────────────────────────
        all_tags = {**cfg["mlflow"].get("tags", {}), **tags}
        for key, value in all_tags.items():
            mlflow.set_tag(key, value)

        # ── Params: auto-log everything under cfg["train"] ─────────────
        mlflow.log_params(cfg["train"])

        # ── Metrics ─────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Artifacts: log entire directory at once ─────────────────────
        mlflow.log_artifacts(str(artifact_dir), artifact_path="eval")

        # ── Model: log + register ───────────────────────────────────────
        input_example = X_train.head(5)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        # ── Stage promotion ─────────────────────────────────────────────
        model_version = None
        if model_info.model_uri and registered_model_name:
            client = MlflowClient()
            versions = client.search_model_versions(
                f"name='{registered_model_name}'"
            )
            run_versions = [
                v for v in versions if getattr(v, "run_id", None) == run_id
            ]
            if run_versions:
                run_versions.sort(key=lambda v: int(v.version))
                model_version = run_versions[-1].version

            if model_stage and model_version is not None:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=model_version,
                    stage=model_stage,
                )

        # ── Summary ─────────────────────────────────────────────────────
        print("MLflow run complete")
        print(f"  Run ID:           {run_id}")
        print(f"  Experiment:       {resolved['experiment_name']}")
        print(f"  Registered model: {registered_model_name}")
        if model_version is not None:
            print(f"  Model version:    {model_version}")
        if model_stage and model_version is not None:
            print(f"  Stage '{model_stage}' -> v{model_version}")
