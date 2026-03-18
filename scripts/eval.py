import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation import (
    compute_confusion_matrix,
    compute_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    split_dir = Path(cfg["data"]["split_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    metrics_dir = Path(cfg["artifacts"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(split_dir / "test.csv")

    with (model_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)

    X_test = test_df.drop(columns=["target", "event_timestamp", "patient_id"])
    y_test = test_df["target"]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, preds, probs)
    cm = compute_confusion_matrix(y_test, preds)

    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, metrics_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, probs, metrics_dir / "roc_curve.png")
    plot_pr_curve(y_test, probs, metrics_dir / "pr_curve.png")
    plot_feature_importance(X_test.columns.to_list(), model.feature_importances_, metrics_dir / "feature_importance.png")

    print(f"Wrote evaluation artifacts to: {metrics_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
