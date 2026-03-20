from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlops-examples-tests-mpl")

import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mlops_examples.data.load import build_feature_snapshot
from mlops_examples.modeling.evaluate import evaluate_model
from mlops_examples.modeling.train import train_model
from mlops_examples.tracking import mlflow as tracking_mod
from mlops_examples.utils import sha256_file


@contextlib.contextmanager
def working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class MlflowLoggingTests(unittest.TestCase):
    def test_log_run_emits_expected_tags_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_path = root / "data" / "raw.csv"
            processed_path = root / "data" / "processed.csv"
            split_dir = root / "data" / "splits"
            snapshot_dir = root / "data" / "features" / "current"
            manifest_path = snapshot_dir / "manifest.json"
            model_dir = root / "artifacts" / "model"
            metrics_dir = root / "artifacts" / "metrics"
            config_path = root / "config.yaml"
            uv_lock_path = root / "uv.lock"

            features, labels = load_breast_cancer(return_X_y=True, as_frame=True)
            base_df = features.iloc[:160].copy()
            base_df["target"] = labels.iloc[:160].to_numpy()

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            base_df.to_csv(raw_path, index=False)
            base_df.to_csv(processed_path, index=False)

            split_df = base_df.copy()
            split_df["event_timestamp"] = pd.date_range("2020-01-01", periods=len(split_df), freq="D")
            split_df["patient_id"] = range(1, len(split_df) + 1)
            train_df, test_df = train_test_split(
                split_df,
                test_size=0.25,
                random_state=42,
                stratify=split_df["target"],
            )

            split_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(split_dir / "train.csv", index=False)
            test_df.to_csv(split_dir / "test.csv", index=False)

            config = {
                "mlflow": {
                    "tracking_uri": "http://placeholder/mlflow",
                    "experiment_name": "mlops-examples/test",
                    "registered_model_name": "MLOpsExamples_Test",
                    "tags": {"pipeline": "mlops-examples"},
                },
                "data": {
                    "raw_path": str(raw_path),
                    "processed_path": str(processed_path),
                    "split_dir": str(split_dir),
                },
                "split": {"seed": 42, "test_size": 0.25},
                "train": {
                    "n_estimators": 10,
                    "max_depth": 4,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "seed": 42,
                },
                "features": {
                    "feature_store_path": "feature_store",
                    "feature_service_name": "patient_features",
                    "snapshot_dir": str(snapshot_dir),
                    "manifest_path": str(manifest_path),
                },
                "artifacts": {
                    "model_dir": str(model_dir),
                    "metrics_dir": str(metrics_dir),
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            uv_lock_path.write_text("lockfile", encoding="utf-8")

            with patch("mlops_examples.data.load.get_git_sha", return_value="deadbeef"):
                build_feature_snapshot(str(config_path))

            snapshot_dvc_path = snapshot_dir.with_suffix(".dvc")
            snapshot_dvc_path.write_text(
                "md5: snapshot-md5-value\nouts:\n- path: current\n",
                encoding="utf-8",
            )

            train_model(str(config_path))
            evaluate_model(str(config_path))

            logged_tags: dict[str, str] = {}
            logged_params: list[dict[str, object]] = []
            logged_artifact_calls: list[tuple[str, str]] = []
            logged_artifacts_calls: list[tuple[str, str]] = []

            with (
                patch.object(tracking_mod.mlflow, "set_tracking_uri") as set_tracking_uri,
                patch.object(tracking_mod.mlflow, "set_experiment") as set_experiment,
                patch.object(tracking_mod.mlflow, "start_run", return_value=_FakeRun("run-123")),
                patch.object(
                    tracking_mod.mlflow,
                    "set_tag",
                    side_effect=lambda key, value: logged_tags.__setitem__(key, value),
                ),
                patch.object(
                    tracking_mod.mlflow,
                    "log_params",
                    side_effect=lambda params: logged_params.append(params),
                ),
                patch.object(tracking_mod.mlflow, "log_metrics") as log_metrics,
                patch.object(
                    tracking_mod.mlflow,
                    "log_artifacts",
                    side_effect=lambda path, artifact_path=None: logged_artifacts_calls.append(
                        (path, artifact_path)
                    ),
                ),
                patch.object(
                    tracking_mod.mlflow,
                    "log_artifact",
                    side_effect=lambda path, artifact_path=None: logged_artifact_calls.append(
                        (path, artifact_path)
                    ),
                ),
                patch.object(tracking_mod.mlflow.models, "infer_signature", return_value="sig"),
                patch.object(tracking_mod.mlflow.sklearn, "log_model") as log_model,
                patch.object(tracking_mod, "get_git_sha", return_value="deadbeef"),
                patch.dict(
                    os.environ,
                    {
                        "MLFLOW_TRACKING_URI": "http://override/mlflow",
                        "MLFLOW_EXPERIMENT_NAME": "override-experiment",
                        "MLFLOW_REGISTERED_MODEL_NAME": "Override_Model",
                    },
                    clear=False,
                ),
            ):
                with working_directory(root):
                    tracking_mod.log_run(str(config_path))

            self.assertEqual(set_tracking_uri.call_args.args[0], "http://override/mlflow")
            self.assertEqual(set_experiment.call_args.args[0], "override-experiment")
            self.assertEqual(log_metrics.call_args.args[0]["test_accuracy"] > 0, True)
            self.assertEqual(logged_tags["git_sha"], "deadbeef")
            self.assertEqual(logged_tags["feature_service_name"], "patient_features")
            self.assertEqual(logged_tags["feature_snapshot_id"], json.loads(manifest_path.read_text())["feature_snapshot_id"])
            self.assertEqual(logged_tags["feature_snapshot_dvc_hash"], "snapshot-md5-value")
            self.assertEqual(logged_tags["raw_data_sha256"], sha256_file(raw_path))
            self.assertEqual(logged_tags["processed_data_sha256"], sha256_file(processed_path))
            self.assertIn(str(metrics_dir), {path for path, _ in logged_artifacts_calls})
            self.assertIn((str(config_path), "config"), logged_artifact_calls)
            self.assertIn((str(manifest_path), "lineage"), logged_artifact_calls)
            self.assertIn(("uv.lock", "environment"), logged_artifact_calls)
            self.assertEqual(log_model.call_args.kwargs["registered_model_name"], "Override_Model")
            self.assertEqual(len(logged_params), 2)
