import hashlib
import os
import subprocess
from pathlib import Path
import pandas as pd
import yaml
import random
import numpy as np
import torch


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_df(df: pd.DataFrame) -> str:
    # Stable hash across runs: sort columns, reset index, hash rows
    normalized = df.sort_index(axis=1).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy()
    h = hashlib.sha256()
    h.update(row_hashes.tobytes())
    return h.hexdigest()

def get_git_sha() -> str:
    """Return the current Git commit SHA.

    Checks common CI environment variables first, then falls back to
    ``git rev-parse HEAD``.  Returns ``"unknown"`` if neither is available.
    """
    for key in ("CI_COMMIT_SHA", "GIT_COMMIT", "GITHUB_SHA"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["data_dir"] = Path(cfg["data_dir"])
    cfg["real_test_dir"] = Path(cfg["real_test_dir"])
    cfg["snapshots_dir"] = Path(cfg["snapshots_dir"])
    return cfg

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)