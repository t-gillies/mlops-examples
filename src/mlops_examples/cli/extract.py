from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/breast_cancer.csv")
    parser.add_argument("--append-row", action="store_true")
    parser.add_argument("--seed-mode", choices=["hash", "seed"], default="hash")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from mlops_examples.data.extract import extract_dataset

    extract_dataset(args.out, args.append_row, args.seed_mode, args.seed)


if __name__ == "__main__":
    main()
