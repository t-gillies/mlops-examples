from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlops_examples.config import load_config


def train_model(config_path: str) -> None:
    cfg = load_config(config_path)

    split_dir = Path(cfg["data"]["split_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(split_dir / "train.csv")
    x_train = train_df.drop(columns=["target", "event_timestamp", "patient_id"])
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
    model.fit(x_train, y_train)

    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    print(f"Wrote model to: {model_path.resolve()}")
