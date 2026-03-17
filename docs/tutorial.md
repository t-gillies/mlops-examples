# MLOps Examples Tutorial: Reproducible Training + Registry

This tutorial walks through a repeatable workflow:
1. Create a new branch
2. Create a new DVC data version
3. Store features
4. Train and register a model
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
git checkout -b demo/repro-1
```

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
Training pulls features from the Feast offline store (Postgres). The recommended workflow is to run Feast from the docker runner on the shared `mlops` network.

Apply the Feast definitions:
```bash
make runner-build
make load-docker
```

Training uses the FeatureService `patient_features`. Re-run `feast apply` if
you change feature definitions.

---

## Step 4: Train + Register a Model
```bash
make train-docker
```

Verify in MLflow UI:
- Experiment: `mlops-examples/dev`
- A new run exists with metrics and artifacts
- A model is registered: `MLOpsExamples_BreastCancer_RF`
- Run tags include `git_sha`, `data_sha256`, and `feature_sha256`

Artifacts in `eval/`:
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

Run training each time:
```bash
make train-docker
```

In MLflow, compare the runs:
- `val_accuracy`, `val_f1_macro`, `val_precision`, `val_recall`
- `val_roc_auc`, `val_pr_auc`
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
make load-docker
make train-docker
```

---

## Step 7: Repeat the Data-Version Process
Run the same steps to generate a new data version and compare results.

```bash
make extract-append
git add data/raw/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Update dataset version"
make push
make load-docker
make train-docker
```

In MLflow, compare the two runs:
- `val_accuracy`, `val_f1_macro`, `val_precision`, `val_recall`
- `val_roc_auc`, `val_pr_auc`
- Compare confusion/ROC/PR/feature importance plots

---

## Step 8: Reproduce a Registry Model Version
To reproduce a registered model:

1) In MLflow Registry, open the model version and click the run.

2) Note the run tags:
   - `git_sha`
   - `data_sha256`
   - `feature_sha256`
   
3) Reproduce locally:
```bash
git checkout <git_sha>
make pull
make load-docker
make train-docker
```

You should get matching metrics and the same artifacts for that run.

**Note:** `data_sha256` and `feature_sha256` are verification tags, not command inputs. The `git_sha` points to the commit whose `.dvc` file references the exact data version. If the run was created before committing the `.dvc` file in Step 2, `make pull` cannot restore the data, and reproduction will fail. Use `data_sha256` to cross-check the pulled dataset and `feature_sha256` to confirm the rebuilt Feast dataframe matches the original run.

---

## Step 9: MLflow Features to Explore
Use the MLflow UI to practice core workflows:

1) Compare runs
   - Select two runs from the experiment list.
   - Click “Compare” to view params/metrics side‑by‑side.
   - Observe how metrics shift across data versions.

2) Filter and search
   - Use the filter bar to find runs with `val_roc_auc > 0.97`.
   - Filter by tag `pipeline = mlops-examples`.

3) Inspect artifacts
   - Open `eval/` and compare `roc_curve.png` or `feature_importance.png`.

4) Model Registry
   - Open the registered model `MLOpsExamples_BreastCancer_RF`.
   - Inspect versions and their source runs.
   - If stages are enabled, promote a version to `Staging`.

---

## Step 10: Cleanup (Optional)
```bash
git checkout main
git branch -D demo/repro-1
```
