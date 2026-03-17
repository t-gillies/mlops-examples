import argparse
from pathlib import Path
import os

import pandas as pd
import yaml
from sqlalchemy import create_engine


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    processed_data_path = Path(cfg["data"]["processed_path"])

    df = pd.read_csv(processed_data_path)
    if "target" not in df.columns:
        raise ValueError("Dataset must include a 'target' column.")

    # Build features + target
    features_df = df.drop(columns=["target"]).copy()
    target_df = df[["target"]].copy()

    # Add event_timestamp + patient_id
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="D").to_list()
    features_df["event_timestamp"] = timestamps
    target_df["event_timestamp"] = timestamps
    features_df["patient_id"] = list(range(1, len(df) + 1))
    target_df["patient_id"] = list(range(1, len(df) + 1))

    # Write to Postgres Offline Store
    engine = create_engine(os.path.expandvars(cfg["features"]["offline_store_uri"]))
    features_df.to_sql("features_df", con=engine, schema="public", if_exists="replace", index=False)
    target_df.to_sql("target_df", con=engine, schema="public", if_exists="replace", index=False)

    print("Loaded features to offline store:")
    print("  tables: public.features_df, public.target_df")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)