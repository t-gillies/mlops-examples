from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mlops_examples.data import extract as extract_mod


def _base_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 1.2, 2.0, 2.2],
            "feature_b": [10.0, 10.5, 20.0, 20.5],
            "target": [0, 0, 1, 1],
        }
    )


class ExtractTests(unittest.TestCase):
    def test_seed_from_hash_is_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "dataset.csv"
            data_path.write_text("value\n1\n", encoding="utf-8")
            first_seed = extract_mod.seed_from_hash(data_path)
            second_seed = extract_mod.seed_from_hash(data_path)

            data_path.write_text("value\n2\n", encoding="utf-8")
            changed_seed = extract_mod.seed_from_hash(data_path)

        self.assertEqual(first_seed, second_seed)
        self.assertNotEqual(first_seed, changed_seed)

    def test_append_one_row_marks_commit_and_blocks_repeat_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "breast_cancer.csv"
            marker_path = root / "breast_cancer.appended"
            _base_dataset().to_csv(data_path, index=False)

            with (
                patch.object(extract_mod, "MARKER_PATH", marker_path),
                patch.object(extract_mod, "get_git_sha", return_value="abc123"),
            ):
                extract_mod.append_one_row(data_path, seed_mode="seed", seed=7)
                updated_df = pd.read_csv(data_path)

                self.assertEqual(len(updated_df), 5)
                self.assertEqual(marker_path.read_text(encoding="utf-8").strip(), "abc123")

                with self.assertRaisesRegex(RuntimeError, "already appended"):
                    extract_mod.append_one_row(data_path, seed_mode="seed", seed=7)

