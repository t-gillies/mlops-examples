from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_sha() -> str:
    """Return the current Git commit SHA or ``unknown`` when unavailable."""
    for key in ("CI_COMMIT_SHA", "GIT_COMMIT", "GITHUB_SHA"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_dvc_hash(path: Path) -> str | None:
    """Return the top-level DVC content hash from a tracked artifact file."""
    if not path.exists():
        return None

    data = yaml.safe_load(path.read_text()) or {}
    return data.get("md5") or data.get("hash")
