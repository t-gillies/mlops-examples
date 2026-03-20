from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()

    from mlops_examples.modeling.evaluate import evaluate_model

    evaluate_model(args.config)


if __name__ == "__main__":
    main()
