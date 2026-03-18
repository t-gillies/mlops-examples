# CI/CD Walkthrough (High Level)

The pipeline mirrors local steps:

1) **Pull data with DVC**
   - Fetches the version referenced in the repo from the docker runner on the shared network.
2) **Build Feast offline store**
   - Runs the dockerized Feast workflow to populate Postgres tables from the shared `mlops` network.
3) **Create train/val/test splits**
   - Uses Feast offline features and writes split artifacts.
4) **Train the model**
   - Uses the runner container with `configs/ci.yaml`.

The checked-in CI job currently focuses on the dockerized training path. Evaluation and MLflow logging remain available as local workflow steps.

The pipeline now uses the same Make targets as local development:
- `make runner-build`
- `make pull`
- `make load-docker TRAIN_CONFIG=configs/ci.yaml`
- `make split-docker TRAIN_CONFIG=configs/ci.yaml`
- `make train-docker TRAIN_CONFIG=configs/ci.yaml`

`TRAIN_CONFIG` is set as a CI job variable so the dockerized feature-load, split, and train steps all use the CI-specific config.

## Why this matters
CI ensures your pipeline is reproducible in a clean environment.
If it fails in CI, you likely have a missing dependency, bad config, or missing credentials.

## Runner requirement
- The GitLab runner must have Docker access to the same host and `mlops` Docker network as `mlops-services`.
- A shell runner on the shared host is the simplest fit.
- A Docker executor can also work if it has the host Docker socket mounted and can see the shared network.

## Required CI variables
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLOPS_NETWORK` (optional, defaults to `mlops`)
- `PUBLIC_FQDN` (optional, defaults to `mlops-nginx`)
- `MLFLOW_BASE_PATH` (optional, defaults to `mlflow`)
- `MLFLOW_EXPERIMENT_NAME` (optional override)
- `MLFLOW_REGISTERED_MODEL_NAME` (optional override)
- `MLFLOW_MODEL_STAGE` (optional)
