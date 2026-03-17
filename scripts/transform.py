import argparse
from pathlib import Path

import torch

from src.transform import TransformSubset, get_train_transforms, get_val_transforms
from src.utils import load_cfg


def main(config_path: str,dataset_path: str | None = None,in_dir: str | None = None, out_dir: str | None = None) -> None:
    cfg = load_cfg(config_path)

    dataset_path = Path(dataset_path)
    input_dir = Path(in_dir)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading dataset: {dataset_path}")
    full_dataset = torch.load(dataset_path, weights_only=False)

    train_indices = torch.load(input_dir / "train_indices.pt", weights_only=False)
    val_indices = torch.load(input_dir / "val_indices.pt", weights_only=False)
    test_indices = torch.load(input_dir / "test_indices.pt", weights_only=False)

    train_ds = TransformSubset(
        full_dataset,
        train_indices,
        get_train_transforms(cfg),
        use_mirroring=cfg["use_mirroring"],
    )

    val_ds = TransformSubset(
        full_dataset,
        val_indices,
        get_val_transforms(cfg),
        use_mirroring=False,
    )

    test_ds = TransformSubset(
        full_dataset,
        test_indices,
        get_val_transforms(cfg),
        use_mirroring=False,
    )

    train_path = output_dir / "train_ds.pt"
    val_path = output_dir / "val_ds.pt"
    test_path = output_dir / "test_ds.pt"

    torch.save(train_ds, train_path)
    torch.save(val_ds, val_path)
    torch.save(test_ds, test_path)

    print(f"Effective sizes: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"Serialized transformed train dataset to: {train_path}")
    print(f"Serialized transformed val dataset to: {val_path}")
    print(f"Serialized transformed test dataset to: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    main(args.config, args.dataset, args.in_dir, args.out_dir)
