from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlops-examples-tests-mpl")

import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mlops_examples.modeling.evaluate import evaluate_model
from mlops_examples.modeling.train import train_model


class TrainingWorkflowTests(unittest.TestCase):
    def test_train_and_evaluate_write_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "splits"
            model_dir = root / "artifacts" / "model"
            metrics_dir = root / "artifacts" / "metrics"
            config_path = root / "config.yaml"

            features, labels = load_breast_cancer(return_X_y=True, as_frame=True)
            df = features.iloc[:160].copy()
            df["target"] = labels.iloc[:160].to_numpy()
            df["event_timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
            df["patient_id"] = range(1, len(df) + 1)

            train_df, test_df = train_test_split(
                df,
                test_size=0.25,
                random_state=42,
                stratify=df["target"],
            )

            split_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(split_dir / "train.csv", index=False)
            test_df.to_csv(split_dir / "test.csv", index=False)

            config = {
                "data": {"split_dir": str(split_dir)},
                "artifacts": {
                    "model_dir": str(model_dir),
                    "metrics_dir": str(metrics_dir),
                },
                "train": {
                    "n_estimators": 25,
                    "max_depth": 4,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "seed": 42,
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            train_model(str(config_path))
            evaluate_model(str(config_path))

            metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue((model_dir / "model.pkl").exists())
            self.assertEqual(
                set(metrics),
                {
                    "test_accuracy",
                    "test_f1_macro",
                    "test_precision",
                    "test_recall",
                    "test_roc_auc",
                    "test_pr_auc",
                },
            )
            for plot_name in (
                "confusion_matrix.png",
                "roc_curve.png",
                "pr_curve.png",
                "feature_importance.png",
            ):
                self.assertTrue((metrics_dir / plot_name).exists(), plot_name)

