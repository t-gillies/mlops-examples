import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.utils import sha256_file

MARKER_PATH = Path("data/raw/breast_cancer.appended")


def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def seed_from_hash(path: Path) -> int:
    digest = sha256_file(path)
    return int(digest[:16], 16) % (2**32)


def append_one_row(
    data_path: Path, seed_mode: str, seed: int | None
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run `make extract` first.")

    git_sha = get_git_sha()
    if git_sha:
        if MARKER_PATH.exists() and MARKER_PATH.read_text().strip() == git_sha:
            raise RuntimeError(
                "Dataset already appended for this commit. Commit your changes before appending again."
            )
    else:
        print("Warning: git not available; append safeguard disabled.")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Dataset must include a 'target' column.")

    X = df.drop(columns=["target"])
    y = df["target"]

    stats = df.groupby("target")[X.columns].agg(["mean", "std"])
    class_counts = y.value_counts().sort_index()
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

    row = {}
    for col in X.columns:
        mu = float(stats.loc[target, (col, "mean")])
        sigma = float(stats.loc[target, (col, "std")])
        if np.isnan(sigma) or sigma == 0:
            value = mu
        else:
            value = rng.normal(mu, sigma)
            value = float(np.clip(value, mu - sigma, mu + sigma))
        row[col] = value
    row["target"] = target

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(data_path, index=False)
    print(f"Appended 1 row to: {data_path.resolve()}  (rows={len(df)})")

    if git_sha:
        MARKER_PATH.write_text(git_sha + "\n")


def main(out_path: str, append_row: bool, seed_mode: str, seed: int | None) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if append_row:
        append_one_row(out, seed_mode, seed)
        return

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y
    df.to_csv(out, index=False)
    print(f"Wrote dataset to: {out.resolve()}  (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/breast_cancer.csv")
    parser.add_argument("--append-row", action="store_true")
    parser.add_argument("--seed-mode", choices=["hash", "seed"], default="hash")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.out, args.append_row, args.seed_mode, args.seed)
