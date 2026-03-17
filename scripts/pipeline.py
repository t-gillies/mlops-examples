import argparse
from pathlib import Path

from scripts.eval import main as eval_main
from scripts.extract import main as extract_main
from scripts.log import main as log_main
from scripts.split import main as split_main
from scripts.train import main as train_main
from scripts.transform import main as transform_main


def main(config_path: str, data_dir: str | None = None) -> None:
    processed_dir = Path(data_dir) if data_dir else Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "t92_dataset.pt"

    print("=" * 60)
    print("T92 PIPELINE")
    print("=" * 60)

    print("\n[1/5] Extract")
    extract_main(config_path, str(dataset_path))

    print("\n[2/5] Split")
    split_main(config_path, str(dataset_path), str(processed_dir))

    print("\n[3/5] Transform")
    transform_main(config_path, str(dataset_path), str(processed_dir), str(processed_dir))

    print("\n[4/5] Train")
    run_dir = train_main(config_path, str(processed_dir))

    print("\n[5/6] Eval")
    eval_main(config_path, str(run_dir), str(processed_dir / "test_ds.pt"))

    print("\n[6/6] Log")
    log_main(config_path, str(run_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    main(args.config, args.data_dir)
