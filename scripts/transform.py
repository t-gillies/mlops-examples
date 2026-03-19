import argparse
from pathlib import Path

import pandas as pd


def main(input_path: str, output_path: str) -> None:
    src = Path(input_path)
    dst = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {src}")

    df = pd.read_csv(src)

    #Transform Operations

    df = df.drop_duplicates().reset_index(drop=True)

    # End Transform Operations

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)

    print(f"Read:  {src.resolve()}")
    print(f"Wrote: {dst.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/breast_cancer.csv")
    parser.add_argument("--output", default="data/processed/breast_cancer_transformed.csv")
    args = parser.parse_args()
    main(args.input, args.output)