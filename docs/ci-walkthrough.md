# CI/CD Walkthrough (High Level)

The pipeline mirrors local steps:

1) **Pull data with DVC**
   - Fetches the version referenced in the repo.
2) **Build Feast offline store (if using Feast)**
   - Runs `make features` or the CI equivalent to populate Postgres tables.
3) **Train the model**
   - Uses `configs/ci.yaml` to run training.
4) **Log to MLflow**
   - Metrics, artifacts, and model registration happen the same as local.

## Why this matters
CI ensures your pipeline is reproducible in a clean environment.
If it fails in CI, you likely have a missing dependency, bad config, or missing credentials.

## Required CI variables
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `MLFLOW_TRACKING_URI` (if not default)
 - `MLFLOW_EXPERIMENT_NAME` (optional override)
 - `MLFLOW_REGISTERED_MODEL_NAME` (optional override)
 - `MLFLOW_MODEL_STAGE` (optional)
