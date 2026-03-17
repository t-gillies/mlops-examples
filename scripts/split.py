import argparse
import copy
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from src.utils import load_cfg


def main(config_path: str, dataset_path: str | None = None, out_dir: str | None = None) -> None:
    cfg = load_cfg(config_path)

    dataset_path = Path(dataset_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading dataset: {dataset_path}")
    full_dataset = torch.load(dataset_path, weights_only=False)

    total = full_dataset.base_length
    indices = list(range(total))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(cfg["val_split"] + cfg["test_split"]),
        random_state=cfg["seed"],
    )

    val_ratio = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio),
        random_state=cfg["seed"],
    )

    train_path = output_dir / "train_indices.pt"
    val_path = output_dir / "val_indices.pt"
    test_path = output_dir / "test_indices.pt"

    torch.save(train_indices, train_path)
    torch.save(val_indices, val_path)
    torch.save(test_indices, test_path)

    print(f"\nBase split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    print(f"Serialized train indices to: {train_path}")
    print(f"Serialized val indices to: {val_path}")
    print(f"Serialized test indices to: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    main(args.config, args.dataset, args.out_dir)
