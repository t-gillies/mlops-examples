import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from feast import FeatureStore
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    split_dir = Path(cfg["data"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)


    store = FeatureStore(repo_path=cfg["features"]["feast_repo"])
    service = store.get_feature_service("patient_features")

    offline_store_uri = os.path.expandvars(cfg["features"]["offline_store_uri"])
    engine = create_engine(offline_store_uri)
    entity_df = pd.read_sql("SELECT * FROM public.target_df", con=engine)

    df = store.get_historical_features(
        entity_df=entity_df,
        features=service
    ).to_df()



#################################

    if "target" not in df.columns:
        raise ValueError("Retrieved dataset must include a 'target' column.")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["seed"],
        stratify=df["target"],
    )

    #Uncomment if you need validation set

    # val_ratio = cfg["split"]["val_size"] / (
    #     1.0 - cfg["split"]["test_size"]
    # )

    # train_df, val_df = train_test_split(
    #     train_df,
    #     test_size=val_ratio,
    #     random_state=int(cfg["split"]["seed"]),
    #     stratify=train_df["target"],
    # )

    train_df.to_csv(split_dir / "train.csv", index=False)
    # val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    print(f"Wrote splits to: {split_dir.resolve()}")
    print(f"  train: {len(train_df)} rows")
    # print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
