# Failure Modes Exercise

Practice diagnosing common issues in a safe, repeatable way.

## 1) Wrong AWS credentials (DVC pull fails)
Break it:
```bash
export AWS_ACCESS_KEY_ID="wrong"
export AWS_SECRET_ACCESS_KEY="wrong"
make pull
```

Fix it:
```bash
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
make pull
```

## 2) MLflow tracking URI unreachable
Break it:
```bash
export MLFLOW_TRACKING_URI="http://localhost:5999"
make train
```

Fix it:
```bash
unset MLFLOW_TRACKING_URI
make train
```

## 3) DVC remote endpoint misconfigured
Break it (temporarily):
```bash
sed -n '1,120p' .dvc/config
```
Edit the `endpointurl` to a bad value, then try:
```bash
make pull
```

Fix it:
Revert the config change and re-run:
```bash
make pull
```

## 4) Feast offline store unreachable
Break it:
```bash
export POSTGRES_HOST="bad-host"
make features
```

Fix it:
```bash
unset POSTGRES_HOST
make features
```

### What to observe
- Errors should point to credentials or network connectivity.
- Ensure fixes are done via env vars or config, not by editing code.
