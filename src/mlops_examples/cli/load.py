from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()

    from mlops_examples.data.load import load_features

    load_features(args.config)


if __name__ == "__main__":
    main()
