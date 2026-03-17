# MLOps Examples: DVC + MLflow + Model Registry (On-Prem)

This repo is a teaching + demo project for our MLOps stack:

- DVC: data versioning (Git tracks metadata; data lives in object storage)
- MLflow: experiment tracking + artifacts + Model Registry
- RustFS (S3-compatible): artifact storage for DVC (and MLflow server artifacts)
- Feast: feature store for training data
- Postgres: MLflow backend store (runs/registry metadata) and Feast offline store

For a full walkthrough, see `docs/tutorial.md`. See `docs/README.md` for the full docs index.

## Prereqs
- Python 3.11+
- `uv` (required)
- Git
- DVC remote and MLflow tracking server reachable from your machine/network
- Docker, if you want to use the recommended runner-container workflow for Feast + training

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
- MLflow UI: http://localhost/mlflow
- RustFS Console: http://localhost/rustfs

## Makefile commands

Run `make help` to see the command list in your terminal.

- `make setup`: Create `.venv` and install dependencies via `uv sync`
- `make lock`: Refresh `uv.lock`
- `make extract`: Generate deterministic dataset and track it with DVC
- `make extract-append`: Append one row to the dataset and track it with DVC
- `make pull`: Recommended DVC pull behind the shared runner/container workflow
- `make push`: Recommended DVC push behind the shared runner/container workflow
- `make pull-host`: Host-based DVC pull if a direct S3 endpoint is reachable
- `make push-host`: Host-based DVC push if a direct S3 endpoint is reachable
- `make pull-docker`: Alias for `make pull`
- `make push-docker`: Alias for `make push`
- `make runner-build`: Build/update the runner image used for dockerized jobs
- `make transform`: Transform the raw dataset into processed data
- `make load-docker`: Apply Feast definitions and load features into Postgres from the runner container
- `make train-docker`: Train and log with `configs/dev.yaml` from the runner container
- `make load`: Host-based Feast workflow if Postgres is reachable directly from your machine
- `make train`: Host-based training workflow if Postgres is reachable directly from your machine

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

This repo includes a committed DVC remote pointing at RustFS.

- bucket: `dvc-remote`
- prefix: `mlops-examples`
- committed host fallback endpoint: `http://localhost:9000`

If you need to change these, update `.dvc/config`. The supported dockerized DVC workflow rewrites the endpoint inside the runner container to `http://mlflow-rustfs:9000`, so the committed `localhost` value mainly matters for intentional host-based fallback use.

Important:
- The supported workflow is `make pull` / `make push`, which run DVC inside the runner container and talk to the internal RustFS service on the shared Docker network.
- `make pull-host` / `make push-host` remain available only if you intentionally expose a direct S3-compatible endpoint outside the Docker network.

Credentials are NOT committed. Set these in your shell (or use your secrets manager):

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

`make pull` / `make push` only need RustFS credentials. MLflow credentials are only required for `make load-docker` and `make train-docker`.

If you’re using `mlops-services`, the default DVC targets already load the RustFS credentials from:

```bash
../mlops-services/env/config.env
../mlops-services/env/secrets.env
```

Makefile note: the DVC targets source from `../mlops-services` via `MLOPS_SERVICES_DIR`. If your repo layout differs, override it, e.g.:

```bash
make pull MLOPS_SERVICES_DIR=/path/to/mlops-services
```

### 3) Personal MLflow credentials

Use your own MLflow user account when training. Do not use the bootstrap admin account from `mlops-services`.

Create a local `.env.user` from the example file:

```bash
cp .env.user.example .env.user
# then edit .env.user with your own MLflow username/password
```

You can also export the variables in your shell instead:

```bash
export MLFLOW_TRACKING_USERNAME="your-mlflow-username"
export MLFLOW_TRACKING_PASSWORD="your-mlflow-password"
```

The docker-based Make targets automatically load:
- `../mlops-services/env/config.env`
- `../mlops-services/env/secrets.env`
- `.env.user` if present

Host-based `make load` / `make train` do not source those files automatically. If you use the host fallback, export the needed values yourself, or set `MLFLOW_TRACKING_URI` and the Postgres connection variables explicitly in your shell first.

---

## Create / update the dataset (maintainers)

This project uses a deterministic Breast Cancer CSV to demonstrate DVC.

```bash
uv run python scripts/extract.py --out data/raw/breast_cancer.csv
uv run dvc add data/raw/breast_cancer.csv
git add data/raw/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Track breast_cancer.csv with DVC"
make push
```

If you’ve already run the one-time setup, just use `uv run ...` — no activation required.

Makefile equivalent:

```bash
make extract
git add data/raw/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Track breast_cancer.csv with DVC"
make push
```

After this, others can run `make pull` to fetch the dataset from RustFS with the supported containerized workflow.

---

## Run locally (developer workflow)

### 1) Get the dataset

```bash
make pull
```

### 2) Transform the dataset

```bash
make transform
```

### 3) Load features into the Feast offline store

Training reads features from Feast's offline store in Postgres. With the current `mlops-services` architecture, the recommended workflow is to run Feast from the docker runner on the shared `mlops` network:

```bash
make runner-build
make load-docker
```

If you prefer to run Feast directly on the host and your Postgres instance is reachable from your machine, you can still use:

```bash
make load
```

Host fallback note:
- `make load` expects the Postgres variables referenced by `configs/*.yaml` to already be exported in your shell, such as `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`.

### 4) Train + log to MLflow + register a model

```bash
make train-docker
```

If Postgres is reachable directly from your machine, the host-based fallback remains available:

```bash
make train
```

Host fallback note:
- `make train` also expects the MLflow connection values to already be exported in your shell, either through `MLFLOW_TRACKING_URI` or through `PUBLIC_FQDN` + `MLFLOW_BASE_PATH`, plus your `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`.

Open MLflow UI and verify:

- Experiment exists (`mlops-examples/dev`)
- A run has params + metrics + eval artifacts
- A model is registered (`MLOpsExamples_BreastCancer_RF`)
- Stage promotion (if enabled) is set (e.g. `Staging` -> latest version)
- Run tags include `git_sha`, `data_sha256`, and `feature_sha256`

What to look for in MLflow (Artifacts → `eval/`):
- `metrics.json`: all metrics in one file
- `confusion_matrix.png`: class‑level error breakdown
- `roc_curve.png`: overall separability (AUC in metrics)
- `pr_curve.png`: precision vs recall tradeoff (PR‑AUC in metrics)
- `feature_importance.png`: top predictors from the random forest

---

## GitLab CI

CI runs:

- `make runner-build`
- `make pull`
- `make load-docker TRAIN_CONFIG=configs/ci.yaml`
- `make train-docker TRAIN_CONFIG=configs/ci.yaml`

Runner requirement:
- The GitLab runner must have Docker access to the same host and `mlops` Docker network as `mlops-services`.
- In practice, that means a shell runner on the shared host or a runner with the host Docker socket mounted.

GitLab CI/CD variables required (masked/protected):

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- optional `MLOPS_NETWORK` (defaults to `mlops`)
- optional `PUBLIC_FQDN` (defaults to `mlops-nginx` for CI runner networking)
- optional `MLFLOW_BASE_PATH` (defaults to `mlflow`)
- optional `MLFLOW_EXPERIMENT_NAME` (overrides config)
- optional `MLFLOW_REGISTERED_MODEL_NAME` (overrides config)
- optional `MLFLOW_MODEL_STAGE` (e.g., `Staging`)

---

## Teaching checklist

A) First successful run
- [ ] Clone repo, install deps
- [ ] Set MLflow creds
- [ ] `make pull`
- [ ] `make load-docker`
- [ ] `make train-docker`
- [ ] Find your run in MLflow UI and inspect:
  - params (n_estimators, max_depth, min_samples_leaf, max_features, seed)
  - metrics (val_accuracy, val_f1_macro, val_precision, val_recall, val_roc_auc, val_pr_auc)
  - artifacts (metrics.json, confusion_matrix.png, roc_curve.png, pr_curve.png, feature_importance.png)

B) Prove reproducibility
- [ ] Note the run's `git_sha` and `data_sha256` tags in MLflow
- [ ] Check out that exact git commit
- [ ] `make pull`
- [ ] Re-run `make load-docker` and `make train-docker`, then compare metrics

C) Make a controlled change
- [ ] Change `n_estimators` or `max_depth` in `configs/dev.yaml`
- [ ] Re-run `make train-docker`
- [ ] Compare runs in MLflow (metrics shift, params differ)

D) Data versioning exercise
- [ ] Regenerate data (or add a tiny perturbation in `extract.py` like shuffling rows)
- [ ] `make extract` (or `make extract-append`)
- [ ] `uv run dvc add data/raw/breast_cancer.csv`, commit, `make push`
- [ ] `make load-docker` then `make train-docker` and observe:
  - new `data_sha256` tag
  - potential metric differences
- [ ] Check out the previous commit, `make pull`, rerun `make load-docker` and `make train-docker`, and confirm you can reproduce the old run.

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
  - `feature_sha256`: hash of the Feast training dataframe used by the run
- Buckets:
  - `dvc-remote` for DVC (created by `mlops-services` RustFS init)
  - `mlflow-artifacts` for MLflow server

Note: `uv.lock` is committed to pin exact versions for reproducibility.

If you run into issues, check:

- network access to MLflow/RustFS
- AWS creds for DVC
- that the DVC remote endpointurl is correct for your chosen workflow
