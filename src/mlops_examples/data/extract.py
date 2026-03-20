from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlops_examples.utils import get_git_sha, sha256_file

MARKER_PATH = Path("data/raw/breast_cancer.appended")


def seed_from_hash(path: Path) -> int:
    digest = sha256_file(path)
    return int(digest[:16], 16) % (2**32)


def append_one_row(data_path: Path, seed_mode: str, seed: int | None) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run `make extract` first.")

    git_sha = get_git_sha()
    if git_sha != "unknown":
        if MARKER_PATH.exists() and MARKER_PATH.read_text().strip() == git_sha:
            raise RuntimeError(
                "Dataset already appended for this commit. Commit your changes before appending again."
            )
    else:
        print("Warning: git not available; append safeguard disabled.")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Dataset must include a 'target' column.")

    features = df.drop(columns=["target"])
    labels = df["target"]

    stats = df.groupby("target")[features.columns].agg(["mean", "std"])
    class_counts = labels.value_counts().sort_index()
    class_probs = (class_counts / class_counts.sum()).values
    classes = class_counts.index.to_list()

    if seed_mode == "hash":
        rng_seed = seed_from_hash(data_path)
    else:
        if seed is None:
            raise ValueError("--seed is required when --seed-mode=seed")
        rng_seed = int(seed)

    rng = np.random.default_rng(rng_seed)
    target = rng.choice(classes, p=class_probs)

    row: dict[str, float | int] = {}
    for column in features.columns:
        mu = float(stats.loc[target, (column, "mean")])
        sigma = float(stats.loc[target, (column, "std")])
        if np.isnan(sigma) or sigma == 0:
            value = mu
        else:
            value = float(rng.normal(mu, sigma))
            value = float(np.clip(value, mu - sigma, mu + sigma))
        row[column] = value
    row["target"] = int(target)

    updated_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    updated_df.to_csv(data_path, index=False)
    print(f"Appended 1 row to: {data_path.resolve()}  (rows={len(updated_df)})")

    if git_sha != "unknown":
        MARKER_PATH.write_text(git_sha + "\n")


def extract_dataset(out_path: str, append_row: bool, seed_mode: str, seed: int | None) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append_row:
        append_one_row(output_path, seed_mode, seed)
        return

    features, labels = load_breast_cancer(return_X_y=True, as_frame=True)
    df = features.copy()
    df["target"] = labels
    df.to_csv(output_path, index=False)
    print(f"Wrote dataset to: {output_path.resolve()}  (rows={len(df)})")
