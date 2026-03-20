from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/breast_cancer.csv")
    parser.add_argument("--output", default="data/processed/breast_cancer_transformed.csv")
    args = parser.parse_args()

    from mlops_examples.data.transform import transform_dataset

    transform_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
