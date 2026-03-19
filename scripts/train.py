import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    split_dir = Path(cfg["data"]["split_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(split_dir / "train.csv")

    X_train = train_df.drop(columns=["target", "event_timestamp", "patient_id"])
    y_train = train_df["target"]


    model = RandomForestClassifier(
        n_estimators=int(cfg["train"]["n_estimators"]),
        max_depth=(
            None
            if str(cfg["train"]["max_depth"]).lower() in ("none", "null")
            else int(cfg["train"]["max_depth"])
        ),
        min_samples_split=int(cfg["train"]["min_samples_split"]),
        min_samples_leaf=int(cfg["train"]["min_samples_leaf"]),
        max_features=cfg["train"]["max_features"],
        random_state=int(cfg["train"]["seed"]),
        n_jobs=None,
    )
    model.fit(X_train, y_train)

    with (model_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)


    print(f"Wrote model to: {(model_dir / 'model.pkl').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
