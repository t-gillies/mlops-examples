# Reproducibility Checklist

To reproduce a model run, you need:

- **Code**: the exact Git commit (`git_sha`)
- **Data**: the exact data version (`data_sha256` + DVC remote)
- **Features**: the exact Feast training dataframe (`feature_sha256`)
- **Config**: the training config used for the run
- **Environment**: Python + dependency versions (`uv.lock`)
- **MLflow run**: the run ID to inspect artifacts and metrics
- **Feast offline store**: the feature tables used for training (rebuild with `make load-docker`)

## Quick Repro Steps
1) In MLflow, open the run and note `git_sha`, `data_sha256`, and `feature_sha256`.
2) Checkout that code:
```bash
git checkout <git_sha>
```
3) Pull the exact data:
```bash
make pull
```
4) Rebuild the Feast offline store (if using Feast):
```bash
make load-docker
```
5) Re-run training:
```bash
make train-docker
```
6) Compare metrics/artifacts in MLflow and confirm the rebuilt run matches the original `feature_sha256`.
