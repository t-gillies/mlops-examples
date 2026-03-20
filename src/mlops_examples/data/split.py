from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split

from mlops_examples.config import load_config


def create_splits(config_path: str) -> None:
    cfg = load_config(config_path)

    split_dir = Path(cfg["data"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)

    store = FeatureStore(repo_path=cfg["features"]["feature_store_path"])
    service_name = cfg["features"]["feature_service_name"]
    service = store.get_feature_service(service_name)

    snapshot_dir = Path(cfg["features"]["snapshot_dir"])
    entity_df = pd.read_parquet(snapshot_dir / "targets.parquet")

    df = store.get_historical_features(entity_df=entity_df, features=service).to_df()

    if "target" not in df.columns:
        raise ValueError("Retrieved dataset must include a 'target' column.")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["seed"],
        stratify=df["target"],
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    print(f"Wrote splits to: {split_dir.resolve()}")
    print(f"  feature service: {service_name}")
    print(f"  train: {len(train_df)} rows")
    print(f"  test:  {len(test_df)} rows")
