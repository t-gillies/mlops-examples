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
export PUBLIC_FQDN="localhost:5999"
make log
```

Fix it:
```bash
unset PUBLIC_FQDN
make log
```

## 3) DVC remote endpoint misconfigured
Break it (temporarily):
```bash
sed -n '1,120p' .dvc/config
```
Edit the `endpointurl` to a bad value, then try the host fallback:
```bash
make pull-host
```

Why `pull-host`:
- `make pull` rewrites the DVC endpoint inside the runner container to `http://mlflow-rustfs:9000`, so the committed `.dvc/config` endpoint is only exercised by the host fallback path.

Fix it:
Revert the config change and re-run:
```bash
make pull-host
```

## 4) Feast offline store unreachable
Break it:
```bash
export POSTGRES_HOST="bad-host"
make load-docker
```

Fix it:
```bash
unset POSTGRES_HOST
make load-docker
```

### What to observe
- Errors should point to credentials or network connectivity.
- DVC and Feast run from the docker runner on the shared `mlops` network, while evaluation and MLflow logging happen in separate local steps.
- Ensure fixes are done via env vars or config, not by editing code.
