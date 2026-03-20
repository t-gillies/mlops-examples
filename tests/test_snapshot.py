from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from mlops_examples.data.load import build_feature_snapshot
from mlops_examples.utils import sha256_file


class SnapshotTests(unittest.TestCase):
    def test_build_feature_snapshot_writes_manifest_and_parquet_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_path = root / "processed.csv"
            snapshot_dir = root / "features" / "current"
            manifest_path = snapshot_dir / "manifest.json"
            config_path = root / "config.yaml"

            processed_df = pd.DataFrame(
                {
                    "mean radius": [10.1, 11.2, 12.3],
                    "mean texture": [1.1, 1.2, 1.3],
                    "target": [0, 1, 0],
                }
            )
            processed_df.to_csv(processed_path, index=False)

            config = {
                "data": {"processed_path": str(processed_path)},
                "features": {
                    "snapshot_dir": str(snapshot_dir),
                    "manifest_path": str(manifest_path),
                    "feature_service_name": "patient_features",
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch("mlops_examples.data.load.get_git_sha", return_value="deadbeef"):
                build_feature_snapshot(str(config_path))

            features_df = pd.read_parquet(snapshot_dir / "features.parquet")
            targets_df = pd.read_parquet(snapshot_dir / "targets.parquet")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            expected_snapshot_id = hashlib.sha256(
                f"deadbeef:{sha256_file(processed_path)}:patient_features".encode()
            ).hexdigest()[:16]

            self.assertEqual(
                list(features_df.columns),
                ["mean radius", "mean texture", "event_timestamp", "patient_id"],
            )
            self.assertEqual(
                list(targets_df.columns),
                ["target", "event_timestamp", "patient_id"],
            )
            self.assertEqual(features_df["patient_id"].tolist(), [1, 2, 3])
            self.assertEqual(targets_df["patient_id"].tolist(), [1, 2, 3])
            self.assertEqual(manifest["feature_snapshot_id"], expected_snapshot_id)
            self.assertEqual(manifest["row_count"], 3)
            self.assertEqual(manifest["git_sha"], "deadbeef")

