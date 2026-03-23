# MLOps Examples Tutorial: Reproducible Training + Registry

This tutorial walks through a repeatable workflow:
1. Create a new branch
2. Create a new DVC data version
3. Store features
4. Split, train, evaluate, and log a model
5. Commit the change
6. Repeat to show reproducibility and change tracking

It assumes you are using `mlops-services` locally (MLflow + RustFS + Postgres).

---

## Prerequisites
- Python 3.11+
- `uv` installed
- Git
- `mlops-services` running locally


If your repo layout differs, set the path for Make:
```bash
make pull MLOPS_SERVICES_DIR=/path/to/mlops-services
```

---

## One-Time Setup (Per Machine)
```bash
uv venv
uv sync
```

If you already did this, you can skip it. Use `uv run ...` for all commands.

---

## Step 0: Start Services & Set Credentials
In another terminal:
```bash
cd ../mlops-services
make up
```

Verify:
- MLflow UI: 'http://localhost/mlflow'
- RustFS Console: 'http://localhost/rustfs'

Use the dockerized DVC workflow in this repo. DVC object operations use the internal RustFS service from the runner container.

Set up your personal MLflow credentials. Do not use the bootstrap admin account:

```bash
cp .env.user.example .env.user
# edit .env.user and set your own MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
```

---

## Step 1: Create a New Branch
```bash
git checkout -b demo/<your-name>/repro-1
```

Use a unique branch name for your own walkthrough. The branch in this tutorial is meant to be a personal, disposable workspace for recording the demo commits, not a shared branch name that everyone pushes to.

Examples:
```bash
git checkout -b demo/brent/repro-1
git checkout -b demo/alice/repro-1
```

If you are only running the tutorial locally, the exact name does not matter. If you do push branches to a shared remote, keep the personal prefix so multiple people can run the same tutorial without colliding.

---

## Step 2: Create a New Data Version (DVC)
This generates the Breast Cancer dataset and tracks it with DVC.

**Note:** Commit before training so the run's `git_sha` can reproduce the exact data version; push after commit so the remote data upload matches a recorded Git commit.

```bash
make extract
git add data/raw/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Track breast_cancer dataset"
make push
```

---

## Step 3: Configure Feast Feature Store
Training now pulls features from a DVC-tracked Parquet snapshot. Feast still keeps its registry in Postgres, but the offline training data is read from local Parquet files restored by DVC.

Build and track the feature snapshot:
```bash
make runner-build
make snapshot-docker
git add data/features/current.dvc data/features/.gitignore
git commit -m "Track feature snapshot"
make push
```

`make snapshot-docker` writes DVC metadata under `data/features/`. Commit `data/features/current.dvc` and `data/features/.gitignore`; the Parquet snapshot payload under `data/features/current/` remains ignored and is stored through DVC.

Training uses the FeatureService `patient_features`. The split step re-applies the Feast repo before retrieval so the registry stays in sync with the checked-out code. Feast may also create local SQLite state under `feature_store/data/`; that file is local runtime state and should remain untracked.

---

## Step 4: Split, Train, Evaluate, and Log a Model
The incorporated pipeline now separates data splitting, training, evaluation, and MLflow logging into explicit steps.

Create the train/test splits:
```bash
make split-docker
```

Train the model:
```bash
make train
```

Generate local evaluation artifacts:
```bash
make eval
```

Log params, metrics, artifacts, and register the model in MLflow:
```bash
make log
```

If you want to run the full flow from a clean starting point in one command, use:
```bash
make pipeline
```

Verify in MLflow UI:
- Experiment: `mlops-examples/dev`
- A new run exists with metrics and artifacts
- A model is registered: `MLOpsExamples_BreastCancer_RF`
- Run tags include `git_sha`, `raw_data_sha256`, `processed_data_sha256`, and `feature_snapshot_id`

Artifacts in `metrics/`:
- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `feature_importance.png`

---

## Step 5: Try a Few Hyperparameter Variations
To see performance shifts, change a few Random Forest hyperparameters and re-run training.

Edit `configs/dev.yaml` and try combinations like:
- `n_estimators`: 50, 200, 500
- `max_depth`: 3, 6, 12
- `min_samples_leaf`: 1, 2, 5
- `max_features`: "sqrt" or "log2"

Run the pipeline stages each time:
```bash
make train
make eval
make log
```

In MLflow, compare the runs:
- `test_accuracy`, `test_f1_macro`, `test_precision`, `test_recall`
- `test_roc_auc`, `test_pr_auc`
- ROC/PR curves and feature importance

Repeat a couple of times to get a feel for how model capacity affects performance.

---

## Step 6: Create + Train on a New Data Version
This appends one synthetic row based on per‑class mean/stddev. The random seed is derived from the current dataset hash, so the new row is deterministic for the current dataset.
`data/raw/breast_cancer.appended` is a local guard file used to prevent multiple appends per commit; it is intentionally not tracked.

```bash
make extract-append
git status --short
git add data/raw/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Add breast_cancer dataset version"
make push
make transform
make snapshot-docker
git add data/features/current.dvc data/features/.gitignore
git commit -m "Track updated feature snapshot"
make push
make split-docker
make train
make eval
make log
```

---

## Step 7: Repeat the Data-Version Process
Run the same steps to generate a new data version and compare results.

```bash
make extract-append
git add data/raw/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Update dataset version"
make push
make transform
make snapshot-docker
git add data/features/current.dvc data/features/.gitignore
git commit -m "Track updated feature snapshot"
make push
make split-docker
make train
make eval
make log
```

In MLflow, compare the two runs:
- `test_accuracy`, `test_f1_macro`, `test_precision`, `test_recall`
- `test_roc_auc`, `test_pr_auc`
- Compare confusion/ROC/PR/feature importance plots

---

## Step 8: Reproduce a Registry Model Version
To reproduce a registered model:

1) In MLflow Registry, open the model version and click the run.

2) Note the run tags:
   - `git_sha`
   - `raw_data_sha256`
   - `processed_data_sha256`
   - `feature_snapshot_id`
   
3) Reproduce locally:
```bash
git checkout <git_sha>
make pull
make transform
make snapshot-docker
make split-docker
make train
make eval
make log
```

You should get matching metrics and the same artifacts for that run.

**Note:** The data hashes are verification tags, not command inputs. The `git_sha` points to the commit whose `.dvc` files reference the exact raw data and feature snapshot versions. Use `raw_data_sha256`, `processed_data_sha256`, and `feature_snapshot_id` to verify the restored run lineage.

---

## Step 9: MLflow Features to Explore
Use the MLflow UI to practice core workflows:

1) Compare runs
   - Select two runs from the experiment list.
   - Click “Compare” to view params/metrics side‑by‑side.
   - Observe how metrics shift across data versions.

2) Filter and search
   - Use the filter bar to find runs with `test_roc_auc > 0.97`.
   - Filter by tag `pipeline = mlops-examples`.

3) Inspect artifacts
   - Open `metrics/` and compare `roc_curve.png` or `feature_importance.png`.

4) Model Registry
   - Open the registered model `MLOpsExamples_BreastCancer_RF`.
   - Inspect versions and their source runs.
   - If stages are enabled, promote a version to `Staging`.

---

## Step 10: Run the Same Flow in Airflow
Once the command-line flow works, switch to Airflow to run the same pipeline as a DAG with per-step logs.

This repo includes Airflow DAGs under `dags/`:

- `dags/demo.py`: a small smoke-test DAG with two tasks, `hello -> airflow`
- `dags/mlops_pipeline.py`: the full example pipeline orchestrated in Airflow

Airflow itself runs from `../mlops-services`, but the DAG code lives in this repo so it can evolve alongside the pipeline code and configs.

### What Airflow is doing here
Airflow is the orchestrator. It does not replace DVC, Feast, or MLflow.

Its job is to:

- define the order of the steps
- run the steps
- record which step succeeded or failed
- show task logs in the UI

In this repo, the `mlops_pipeline` DAG mirrors the Makefile flow:

```text
setup_environment
-> extract_data
-> push_data
-> pull_data
-> transform_data
-> snapshot_features
-> push_snapshot
-> split_data
-> train_model
-> evaluate_model
-> log_to_mlflow
```

That is intentionally close to `make pipeline`, but broken into separate Airflow tasks so you can see exactly where a run fails.

### Airflow prerequisites
Before using the DAGs:

1. Start `mlops-services` in another terminal:
```bash
cd ../mlops-services
export MLOPS_EXAMPLES_DIR=/path/to/your/mlops-examples-checkout
make up
```

If your repo layout is the default sibling checkout, use:
```bash
export MLOPS_EXAMPLES_DIR="$(cd ../mlops-examples && pwd)"
```

2. Make sure Airflow is mounting this repo's `dags/` folder from `MLOPS_EXAMPLES_DIR`.

3. Make sure your personal MLflow credentials are available in `.env.user`:
```bash
cp .env.user.example .env.user
# then edit .env.user
```

### Demo DAG
Start with the `demo` DAG. It proves:

- Airflow can discover DAG files
- the scheduler can create a DAG run
- the executor can run tasks
- logs are being written

Run it from the Airflow UI:

1. Open `http://localhost/airflow`
2. Find the `demo` DAG
3. Click the play button to trigger it

Expected behavior:

- task `hello` prints `hello`
- task `airflow` prints `airflow`

### MLOps pipeline DAG
Once the demo DAG works, use `mlops_pipeline`.

Run it from the UI:

1. Open `http://localhost/airflow`
2. Open the `mlops_pipeline` DAG
3. Trigger a run
4. Watch the `Grid` view as each task turns from queued to running to success or failed

This DAG runs the same scripts you already used from the Makefile, but in separate Airflow tasks.

### How to read Airflow failures
When a task fails:

1. Open the DAG in `Grid` view
2. Click the failed task box
3. Open `Log`

When reading the log, use this order:

1. Find the first traceback from your script or command
2. Find the first concrete exception message
3. Treat the large Airflow traceback below it as wrapper context

Examples of root-cause lines:

- `ModuleNotFoundError: No module named 'mlops_examples'`
- `FileNotFoundError: Raw dataset not found at: data/raw/breast_cancer.csv`
- `MlflowException: ... 404 Not Found`

Those lines usually matter more than the long Airflow stack trace underneath.

### Why use Airflow if `make pipeline` already exists?
`make pipeline` is still useful for local command-line runs.

Airflow adds:

- per-step visibility
- DAG run history
- task-level logs
- retries and scheduling later
- a UI for debugging failures

The Makefile is still the simplest mental model for the pipeline.
Airflow is the operational layer that turns that linear workflow into a managed DAG.

### Good beginner workflow
1. Confirm `demo` works
2. Trigger `mlops_pipeline`
3. Watch the `Grid` view
4. If a task fails, open its log and identify the first real exception
5. Fix the issue, then trigger a new run

---

## Step 11: Cleanup (Optional)
```bash
git checkout main
git branch -D demo/<your-name>/repro-1
```
