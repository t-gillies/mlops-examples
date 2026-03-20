from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from mlops_examples.data.split import create_splits


class _FakeHistoricalFeatures:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df


class _FakeFeatureStore:
    dataset: pd.DataFrame | None = None
    instance: "_FakeFeatureStore | None" = None

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.entity_df: pd.DataFrame | None = None
        self.service_name: str | None = None
        _FakeFeatureStore.instance = self

    def get_feature_service(self, name: str) -> str:
        self.service_name = name
        return name

    def get_historical_features(self, entity_df: pd.DataFrame, features: str) -> _FakeHistoricalFeatures:
        self.entity_df = entity_df.copy()
        return _FakeHistoricalFeatures(self.dataset.copy())  # type: ignore[union-attr]


class SplitTests(unittest.TestCase):
    def test_create_splits_reads_targets_snapshot_and_writes_train_test_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            snapshot_dir = root / "data" / "features" / "current"
            split_dir = root / "data" / "processed" / "splits"
            config_path = root / "config.yaml"

            entity_df = pd.DataFrame(
                {
                    "target": [0, 1, 0, 1],
                    "event_timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
                    "patient_id": [1, 2, 3, 4],
                }
            )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            entity_df.to_parquet(snapshot_dir / "targets.parquet", index=False)

            historical_df = pd.DataFrame(
                {
                    "feature_a": [0.1, 0.2, 0.3, 0.4],
                    "target": [0, 1, 0, 1],
                    "event_timestamp": pd.date_range("2020-01-01", periods=4, freq="D"),
                    "patient_id": [1, 2, 3, 4],
                }
            )
            _FakeFeatureStore.dataset = historical_df

            config = {
                "data": {"split_dir": str(split_dir)},
                "features": {
                    "feature_store_path": "feature_store",
                    "feature_service_name": "patient_features",
                    "snapshot_dir": str(snapshot_dir),
                },
                "split": {"seed": 42, "test_size": 0.5},
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch("mlops_examples.data.split.FeatureStore", _FakeFeatureStore):
                create_splits(str(config_path))

            train_df = pd.read_csv(split_dir / "train.csv")
            test_df = pd.read_csv(split_dir / "test.csv")

            self.assertEqual(len(train_df) + len(test_df), len(historical_df))
            self.assertEqual(_FakeFeatureStore.instance.repo_path, "feature_store")
            self.assertEqual(_FakeFeatureStore.instance.service_name, "patient_features")
            pd.testing.assert_frame_equal(_FakeFeatureStore.instance.entity_df, entity_df)

    def test_create_splits_requires_target_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            snapshot_dir = root / "data" / "features" / "current"
            split_dir = root / "data" / "processed" / "splits"
            config_path = root / "config.yaml"

            entity_df = pd.DataFrame(
                {
                    "target": [0, 1],
                    "event_timestamp": pd.date_range("2020-01-01", periods=2, freq="D"),
                    "patient_id": [1, 2],
                }
            )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            entity_df.to_parquet(snapshot_dir / "targets.parquet", index=False)

            _FakeFeatureStore.dataset = pd.DataFrame(
                {
                    "feature_a": [0.1, 0.2],
                    "event_timestamp": pd.date_range("2020-01-01", periods=2, freq="D"),
                    "patient_id": [1, 2],
                }
            )

            config = {
                "data": {"split_dir": str(split_dir)},
                "features": {
                    "feature_store_path": "feature_store",
                    "feature_service_name": "patient_features",
                    "snapshot_dir": str(snapshot_dir),
                },
                "split": {"seed": 42, "test_size": 0.5},
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch("mlops_examples.data.split.FeatureStore", _FakeFeatureStore):
                with self.assertRaisesRegex(ValueError, "include a 'target' column"):
                    create_splits(str(config_path))

