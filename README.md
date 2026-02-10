# MLOps Examples: DVC + MLflow + Model Registry (On-Prem)

This repo is a teaching + demo project for our MLOps stack:

- DVC: data versioning (Git tracks metadata; data lives in object storage)
- MLflow: experiment tracking + artifacts + Model Registry
- RustFS (S3-compatible): artifact storage for DVC (and MLflow server artifacts)
- Postgres: MLflow backend store (runs/registry metadata)

For a full walkthrough, see `docs/tutorial.md`. See `docs/README.md` for the full docs index.

## Prereqs
- Python 3.11+
- `uv` (required)
- Git
- DVC remote and MLflow tracking server reachable from your machine/network

### Install uv
`uv` is a fast Python package manager and virtual environment tool. We prefer it here because it installs quickly, respects the committed `uv.lock` for reproducible environments, and reduces “works on my machine” issues during training.

Use the official installer:

macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

## Local endpoints (mlops-services default)
- MLflow: http://localhost:5000
- RustFS S3: http://localhost:9000
- RustFS Console: http://localhost:9001

## Makefile commands

Run `make help` to see the command list in your terminal.

- `make setup`: Create `.venv` and install dependencies via `uv sync`
- `make lock`: Refresh `uv.lock`
- `make data`: Generate deterministic dataset and track it with DVC
- `make data-append`: Append one row to the dataset and track it with DVC
- `make pull`: Pull dataset from DVC remote (auto-loads creds from `../mlops-services` if needed)
- `make push`: Push dataset to DVC remote (auto-loads creds from `../mlops-services` if needed)
- `make train`: Train and log with `configs/dev.yaml`

---

## One-time setup (per developer machine)

### 1) Install dependencies
This repo requires `uv` for reproducible environments. Python 3.11+ is required.

```bash
uv venv
uv sync
```

Or use the Makefile:

```bash
make setup
```

Notes:
- This is a one-time setup per machine unless you delete `.venv`.
- You do not need to "activate" the venv when using `uv`. Use `uv run ...` and it will use `.venv` automatically.
- If you want to activate manually, run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows PowerShell).

### 2) DVC remote (already configured)

This repo includes a committed DVC remote pointing at RustFS:

- bucket: `dvc-remote`
- prefix: `mlops-examples`
- endpoint: `http://localhost:9000`

If you need to change these, update `.dvc/config`.

Credentials are NOT committed. Set these in your shell (or use your secrets manager):

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

If you’re using `mlops-services`, you can load the defaults and secrets from there:

```bash
set -a
source ../mlops-services/env/config.env
source ../mlops-services/env/secrets.env
set +a
export AWS_ACCESS_KEY_ID="$RUSTFS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$RUSTFS_SECRET_KEY"
```

Makefile note: the default `make pull` / `make push` targets source from `../mlops-services` via `MLOPS_SERVICES_DIR`. If your repo layout differs, override it, e.g.:

```bash
make pull MLOPS_SERVICES_DIR=/path/to/mlops-services
```

---

## Create / update the dataset (maintainers)

This project uses a deterministic Breast Cancer CSV to demonstrate DVC.

```bash
uv run python scripts/make_data.py --out data/breast_cancer.csv
uv run dvc add data/breast_cancer.csv
git add data/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Track breast_cancer.csv with DVC"
uv run dvc push
```

If you’ve already run the one-time setup, just use `uv run ...` — no activation required.

Makefile equivalent:

```bash
make data
git add data/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Track breast_cancer.csv with DVC"
make push
```

After this, others can run `uv run dvc pull` to fetch the dataset from RustFS.

---

## Run locally (developer workflow)

### 1) Get the dataset

```bash
uv run dvc pull
```

If you’ve already run the one-time setup, just use `uv run ...` — no activation required.

Makefile equivalent:

```bash
make pull
```

### 2) Train + log to MLflow + register a model

```bash
uv run python src/train.py --config configs/dev.yaml
```

If you’ve already run the one-time setup, just use `uv run ...` — no activation required.

Makefile equivalent:

```bash
make train
```

Open MLflow UI and verify:

- Experiment exists (`mlops-examples/dev`)
- A run has params + metrics + eval artifacts
- A model is registered (`MLOpsExamples_BreastCancer_RF`)
- Stage promotion (if enabled) is set (e.g. `Staging` -> latest version)

What to look for in MLflow (Artifacts → `eval/`):
- `metrics.json`: all metrics in one file
- `confusion_matrix.png`: class‑level error breakdown
- `roc_curve.png`: overall separability (AUC in metrics)
- `pr_curve.png`: precision vs recall tradeoff (PR‑AUC in metrics)
- `feature_importance.png`: top predictors from the random forest

---

## GitLab CI

CI runs:

- `uv run dvc pull`
- `uv run python src/train.py --config configs/ci.yaml`

GitLab CI/CD variables required (masked/protected):

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- optionally `AWS_DEFAULT_REGION`
- optional `MLFLOW_TRACKING_URI` (overrides config)
- optional `MLFLOW_EXPERIMENT_NAME` (overrides config)
- optional `MLFLOW_REGISTERED_MODEL_NAME` (overrides config)
- optional `MLFLOW_MODEL_STAGE` (e.g., `Staging`)

---

## Teaching checklist

A) First successful run
- [ ] Clone repo, install deps
- [ ] Set AWS creds
- [ ] `dvc pull`
- [ ] Run training (`configs/dev.yaml`)
- [ ] Find your run in MLflow UI and inspect:
  - params (n_estimators, max_depth, min_samples_leaf, max_features, seed)
  - metrics (val_accuracy, val_f1_macro, val_precision, val_recall, val_roc_auc, val_pr_auc)
  - artifacts (metrics.json, confusion_matrix.png, roc_curve.png, pr_curve.png, feature_importance.png)

B) Prove reproducibility
- [ ] Note the run's `git_sha` and `data_sha256` tags in MLflow
- [ ] Check out that exact git commit
- [ ] `uv run dvc pull`
- [ ] Re-run training and compare metrics (should match or be extremely close)

C) Make a controlled change
- [ ] Change `n_estimators` or `max_depth` in `configs/dev.yaml`
- [ ] Re-run training
- [ ] Compare runs in MLflow (metrics shift, params differ)

D) Data versioning exercise
- [ ] Regenerate data (or add a tiny perturbation in `make_data.py` like shuffling rows)
- [ ] `make data` (or `make data-append`)
- [ ] `uv run dvc add data/breast_cancer.csv`, commit, `uv run dvc push`
- [ ] `make train` and observe:
  - new `data_sha256` tag
  - potential metric differences
- [ ] Check out the previous commit, `uv run dvc pull`, rerun and confirm you can reproduce the old run.

E) Registry exercise
- [ ] Identify the latest registered model version
- [ ] If stages are enabled in dev config or env:
  - confirm the model version moved to the configured stage (e.g., `Staging`)
- [ ] Discuss: what would be our policy for promoting `Staging -> Production`?

---

## Notes / conventions

- MLflow tags:
  - `git_sha`: source code version
  - `data_sha256`: dataset content hash
- Buckets:
  - `dvc-remote` for DVC (created by `mlops-services` RustFS init)
  - `mlflow-artifacts` for MLflow server

Note: `uv.lock` is committed to pin exact versions for reproducibility.

If you run into issues, check:

- network access to MLflow/RustFS
- AWS creds for DVC
- that the DVC remote endpointurl is correct
