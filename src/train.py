import argparse
import json
from pathlib import Path

import yaml
from sklearn.ensemble import RandomForestClassifier

from src.data import load_dataset, split_dataset
from src.evaluation import (
    compute_confusion_matrix,
    compute_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)
from src.tracking import log_run
from src.utils import get_git_sha, sha256_file


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    data_path = Path(cfg["data"]["path"])
    out_dir = Path(cfg["artifacts"]["out_dir"])

    # ── Data ────────────────────────────────────────────────────────────
    df = load_dataset(data_path)
    data_hash = sha256_file(data_path)
    git_sha = get_git_sha()

    Xtr, Xte, ytr, yte = split_dataset(
        df,
        test_size=float(cfg["train"]["test_size"]),
        seed=int(cfg["train"]["seed"]),
    )

    # ── Train ───────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=int(cfg["train"]["n_estimators"]),
        max_depth=(
            None
            if str(cfg["train"]["max_depth"]).lower() in ("none", "null")
            else int(cfg["train"]["max_depth"])
        ),
        min_samples_split=int(cfg["train"]["min_samples_split"]),
        min_samples_leaf=int(cfg["train"]["min_samples_leaf"]),
        max_features=cfg["train"]["max_features"],
        random_state=int(cfg["train"]["seed"]),
        n_jobs=None,
    )
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]

    # ── Evaluate ────────────────────────────────────────────────────────
    metrics = compute_metrics(yte, preds, probs)
    cm = compute_confusion_matrix(yte, preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_roc_curve(yte, probs, out_dir / "roc_curve.png")
    plot_pr_curve(yte, probs, out_dir / "pr_curve.png")
    plot_feature_importance(Xtr.columns.to_list(), model.feature_importances_, out_dir / "feature_importance.png")

    # ── Log to MLflow ───────────────────────────────────────────────────
    log_run(
        cfg,
        metrics=metrics,
        artifact_dir=out_dir,
        model=model,
        X_train=Xtr,
        tags={
            "git_sha": git_sha,
            "data_sha256": data_hash,
            "dataset_path": str(data_path),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
