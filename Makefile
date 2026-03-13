.PHONY: help setup lock data data-append pull push features train

help:
	@echo "Make targets:"
	@echo "  make setup         Create virtual environment and install dependencies"
	@echo "  make lock          Refresh uv.lock"
	@echo "  make data          Generate deterministic dataset and track with DVC"
	@echo "  make data-append   Append one row to dataset and track with DVC"
	@echo "  make pull          Pull DVC data (auto-loads creds from mlops-services if needed)"
	@echo "  make push          Push DVC data (auto-loads creds from mlops-services if needed)"
	@echo "  make train         Train model with configs/dev.yaml"

MLOPS_SERVICES_DIR ?= ../mlops-services

define DVC_WITH_ENV
	/bin/bash -lc 'set -euo pipefail; \
	if [ -z "$${AWS_ACCESS_KEY_ID:-}" ] || [ -z "$${AWS_SECRET_ACCESS_KEY:-}" ]; then \
	  CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	  SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	  if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	  if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	  export AWS_ACCESS_KEY_ID="$${AWS_ACCESS_KEY_ID:-$${RUSTFS_ACCESS_KEY:-}}"; \
	  export AWS_SECRET_ACCESS_KEY="$${AWS_SECRET_ACCESS_KEY:-$${RUSTFS_SECRET_KEY:-}}"; \
	fi; \
	uv run dvc $(1)'
endef

setup:
	uv venv
	uv sync

lock:
	uv lock

data:
	uv run python scripts/make_data.py --out data/breast_cancer.csv
	uv run dvc add data/breast_cancer.csv

data-append:
	uv run python scripts/make_data.py --append-row --out data/breast_cancer.csv
	uv run dvc add data/breast_cancer.csv

pull:
	$(call DVC_WITH_ENV,pull)

push:
	$(call DVC_WITH_ENV,push)

features:
	uv run feast -c feature_repo apply
	uv run python scripts/store_features.py --config configs/dev.yaml

train:
	uv run python src/train.py --config configs/dev.yaml
