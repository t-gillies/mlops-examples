import argparse
from pathlib import Path

import torch

from src.utils import load_cfg
from src.extract import T92Dataset


def main(config_path: str, out_path: str | None = None) -> None:
    cfg = load_cfg(config_path)

    print("\nLoading dataset...")
    full_dataset = T92Dataset(
        cfg["data_dir"],
        transform=None,
        use_mirroring=False,
    )

    if len(full_dataset) == 0:
        print("ERROR: No data found!")
        return

    output_path = Path(out_path) if out_path else Path("data/processed/t92_dataset.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_dataset, output_path)

    print(f"Serialized dataset to: {output_path}")
    print(f"Rows: {len(full_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.config, args.out)
