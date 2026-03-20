from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from mlops_examples.config import load_config
from mlops_examples.utils import get_git_sha, sha256_file


def build_feature_snapshot(config_path: str) -> None:
    cfg = load_config(config_path)

    processed_data_path = Path(cfg["data"]["processed_path"])
    snapshot_dir = Path(cfg["features"]["snapshot_dir"])
    manifest_path = Path(cfg["features"]["manifest_path"])
    feature_service_name = cfg["features"]["feature_service_name"]

    df = pd.read_csv(processed_data_path)
    if "target" not in df.columns:
        raise ValueError("Processed dataset must include a 'target' column.")

    features_path = snapshot_dir / "features.parquet"
    targets_path = snapshot_dir / "targets.parquet"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range(start="2020-01-01", periods=len(df), freq="D").to_list()
    patient_ids = list(range(1, len(df) + 1))

    features_df = df.drop(columns=["target"]).copy()
    target_df = df[["target"]].copy()

    features_df["event_timestamp"] = timestamps
    target_df["event_timestamp"] = timestamps
    features_df["patient_id"] = patient_ids
    target_df["patient_id"] = patient_ids

    features_df.to_parquet(features_path, index=False)
    target_df.to_parquet(targets_path, index=False)

    processed_data_sha256 = sha256_file(processed_data_path)
    git_sha = get_git_sha()
    snapshot_id = hashlib.sha256(
        f"{git_sha}:{processed_data_sha256}:{feature_service_name}".encode()
    ).hexdigest()[:16]

    manifest = {
        "feature_service_name": feature_service_name,
        "feature_snapshot_id": snapshot_id,
        "git_sha": git_sha,
        "processed_data_sha256": processed_data_sha256,
        "row_count": len(df),
        "features_path": str(features_path),
        "targets_path": str(targets_path),
        "timestamp_start": str(timestamps[0]) if timestamps else None,
        "timestamp_end": str(timestamps[-1]) if timestamps else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("Wrote feature snapshot:")
    print(f"  feature_service: {feature_service_name}")
    print(f"  snapshot_id:     {snapshot_id}")
    print(f"  features:        {features_path.resolve()}")
    print(f"  targets:         {targets_path.resolve()}")
    print(f"  manifest:        {manifest_path.resolve()}")


def load_features(config_path: str) -> None:
    build_feature_snapshot(config_path)
