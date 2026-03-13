# Stack Overview: DVC + MLflow + RustFS

This repo demonstrates a simple, reproducible MLOps stack:

- **DVC**: tracks dataset versions in Git via `.dvc` files; data lives in object storage.
- **MLflow**: tracks experiments, logs metrics/artifacts, and registers models.
- **RustFS (S3-compatible)**: object store for DVC and MLflow artifacts.
- **Feast**: Feature store for training data.
- **Postgres**: MLflow backend store for runs and registry metadata, and offline feature store.

### How the pieces connect
- DVC stores a content hash and remote location in `data/*.dvc`.
- MLflow logs metrics and artifacts for each run, and records the code/data tags.
- RustFS stores the actual data and artifacts, referenced by both DVC and MLflow.
- Postgres stores MLflow metadata (runs, params, metrics, registry) and features.

### Why this matters
This setup ties **code + data + model** to a specific run. You can always:
1) Checkout the exact Git SHA  
2) Pull the exact data version  
3) Re-run training to reproduce results
