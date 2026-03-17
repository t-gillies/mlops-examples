import argparse
import json
import os
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import (
    compute_confusion_matrix,
    compute_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)
from src.log import log_run
from src.utils import get_git_sha, sha256_file, sha256_df

from sqlalchemy import create_engine
from feast import FeatureStore


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    data_path = Path(cfg["data"]["path"])
    out_dir = Path(cfg["artifacts"]["out_dir"])

    # ── Data ────────────────────────────────────────────────────────────
    
    data_hash = sha256_file(data_path)
    git_sha = get_git_sha()

    # ── Pull Features from Postgres Offline Store ────────────────────────────────────────────────────────────

    store = FeatureStore(repo_path=cfg["features"]["feast_repo"])
    service = store.get_feature_service("patient_features")

    offline_store_uri = os.path.expandvars(cfg["features"]["offline_store_uri"])
    engine = create_engine(offline_store_uri)
    entity_df = pd.read_sql("SELECT * FROM public.target_df", con=engine)

    df = store.get_historical_features(
        entity_df=entity_df,
        features=service
    ).to_df()

    feature_hash = sha256_df(df)

    # ── Split ────────────────────────────────────────────────────────────
    X = df.drop(columns=["target", "event_timestamp", "patient_id"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["train"]["test_size"]),
        random_state=int(cfg["train"]["seed"]),
        stratify=y,
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
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # ── Evaluate ────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test, preds, probs)
    cm = compute_confusion_matrix(y_test, preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, probs, out_dir / "roc_curve.png")
    plot_pr_curve(y_test, probs, out_dir / "pr_curve.png")
    plot_feature_importance(X_train.columns.to_list(), model.feature_importances_, out_dir / "feature_importance.png")

    # ── Log to MLflow ───────────────────────────────────────────────────
    log_run(
        cfg,
        metrics=metrics,
        artifact_dir=out_dir,
        model=model,
        X_train=X_train,
        tags={
            "git_sha": git_sha,
            "data_sha256": data_hash,
            "dataset_path": str(data_path),
            "feature_sha256": feature_hash,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)
