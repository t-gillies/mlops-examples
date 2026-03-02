import hashlib
import os
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
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
