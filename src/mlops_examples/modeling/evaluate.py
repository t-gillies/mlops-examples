from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from mlops_examples.config import load_config
from mlops_examples.modeling.metrics import compute_confusion_matrix, compute_metrics
from mlops_examples.modeling.plots import (
    plot_confusion,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


def evaluate_model(config_path: str) -> None:
    cfg = load_config(config_path)

    split_dir = Path(cfg["data"]["split_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    metrics_dir = Path(cfg["artifacts"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(split_dir / "test.csv")
    with (model_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    x_test = test_df.drop(columns=["target", "event_timestamp", "patient_id"])
    y_test = test_df["target"]

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = compute_metrics(y_test, predictions, probabilities)
    confusion = compute_confusion_matrix(y_test, predictions)

    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(confusion, metrics_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, probabilities, metrics_dir / "roc_curve.png")
    plot_pr_curve(y_test, probabilities, metrics_dir / "pr_curve.png")
    plot_feature_importance(
        x_test.columns.to_list(),
        model.feature_importances_,
        metrics_dir / "feature_importance.png",
    )

    print(f"Wrote evaluation artifacts to: {metrics_dir.resolve()}")
