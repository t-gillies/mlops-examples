from __future__ import annotations

from pathlib import Path

import pandas as pd


def transform_dataset(input_path: str, output_path: str) -> None:
    source_path = Path(input_path)
    destination_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {source_path}")

    df = pd.read_csv(source_path)
    df = df.drop_duplicates().reset_index(drop=True)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination_path, index=False)

    print(f"Read:  {source_path.resolve()}")
    print(f"Wrote: {destination_path.resolve()}")
