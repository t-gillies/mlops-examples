.PHONY: help setup lock data data-append pull push pull-host push-host pull-docker push-docker runner-build features train load-docker train-docker

help:
	@echo "Make targets:"
	@echo "  make setup         Create virtual environment and install dependencies"
	@echo "  make lock          Refresh uv.lock"
	@echo "  make extract          Generate deterministic dataset and track with DVC"
	@echo "  make extract-append   Append one row to dataset and track with DVC"
	@echo "  make pull          Pull DVC data inside docker using the internal RustFS service"
	@echo "  make push          Push DVC data inside docker using the internal RustFS service"
	@echo "  make pull-host     Host-based DVC pull if a direct S3 endpoint is reachable"
	@echo "  make push-host     Host-based DVC push if a direct S3 endpoint is reachable"
	@echo "  make pull-docker   Alias for make pull"
	@echo "  make push-docker   Alias for make push"
	@echo "  make runner-build  Build/update the docker runner image"
	@echo "  make transform      Transform raw data into processed data and track with DVC"
	@echo "  make load      Host-based Feast workflow (requires direct Postgres access)"
	@echo "  make split      Split data into train/val/test sets"
	@echo "  make train         Host-based training workflow (requires direct Postgres access)"
	@echo "  make load-docker  Apply Feast + load offline features inside docker"
	@echo "  make train-docker     Train inside docker on the shared mlops network"
	@echo "  make eval          Run model evaluation and locally create metrics/plots artifacts"
	@echo "  make log           Log training metrics to MLFlow"
	@echo "  make pipeline          Run the full pipeline end-to-end (except for setup and lock)"

	@echo "  Variables: TRAIN_CONFIG=configs/dev.yaml MLOPS_SERVICES_DIR=../mlops-services"

MLOPS_SERVICES_DIR ?= ../mlops-services
TRAIN_CONFIG ?= configs/dev.yaml
RUNNER_COMPOSE = docker compose -f docker-compose.runner.yml
RUNNER_DVC = /opt/venv/bin/dvc
RUNNER_FEAST = /opt/venv/bin/feast
RUNNER_PYTHON = /opt/venv/bin/python

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

define RUNNER_WITH_ENV
	/bin/bash -lc 'set -euo pipefail; \
	CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	USER_ENV=".env.user"; \
	if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	if [ -f "$$USER_ENV" ]; then set -a; source "$$USER_ENV"; set +a; fi; \
	case "$${POSTGRES_HOST:-}" in \
	  "" ) export POSTGRES_HOST="mlflow-postgres" ;; \
	  127.0.0.1|localhost|0.0.0.0 ) export POSTGRES_HOST="mlflow-postgres" ;; \
	esac; \
	: "$${POSTGRES_USER:?Set POSTGRES_USER in your shell or mlops-services env}"; \
	: "$${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in your shell or mlops-services env}"; \
	: "$${MLFLOW_TRACKING_USERNAME:?Set MLFLOW_TRACKING_USERNAME in your shell or .env.user}"; \
	: "$${MLFLOW_TRACKING_PASSWORD:?Set MLFLOW_TRACKING_PASSWORD in your shell or .env.user}"; \
	$(RUNNER_COMPOSE) $(1)'
endef

define DVC_DOCKER_WITH_ENV
	/bin/bash -lc 'set -euo pipefail; \
	CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	USER_ENV=".env.user"; \
	if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	if [ -f "$$USER_ENV" ]; then set -a; source "$$USER_ENV"; set +a; fi; \
	: "$${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID or provide RUSTFS_ACCESS_KEY via mlops-services env}"; \
	: "$${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY or provide RUSTFS_SECRET_KEY via mlops-services env}"; \
	$(RUNNER_COMPOSE) run --rm runner sh -lc '"'"'trap "rm -f .dvc/config.local" EXIT; $(RUNNER_DVC) remote modify --local rustfs endpointurl http://mlflow-rustfs:9000 >/dev/null; $(RUNNER_DVC) $(1)'"'"''
endef

define banner
	@printf '\n============================================================\n'
	@printf ' %s\n' "$(1)"
	@printf '============================================================\n'
endef

setup:
	uv venv
	uv sync

lock:
	uv lock

extract:
	uv run python scripts/extract.py --out data/raw/breast_cancer.csv
	uv run dvc add data/raw/breast_cancer.csv

extract-append:
	uv run python scripts/extract.py --append-row --out data/raw/breast_cancer.csv
	uv run dvc add data/raw/breast_cancer.csv

pull:
	$(call DVC_DOCKER_WITH_ENV,pull)

push:
	$(call DVC_DOCKER_WITH_ENV,push)

pull-host:
	$(call DVC_WITH_ENV,pull)

push-host:
	$(call DVC_WITH_ENV,push)

pull-docker:
	$(call DVC_DOCKER_WITH_ENV,pull)

push-docker:
	$(call DVC_DOCKER_WITH_ENV,push)

runner-build:
	$(call RUNNER_WITH_ENV,build runner)

transform:
	uv run python scripts/transform.py
	uv run dvc add data/processed/breast_cancer_transformed.csv

load:
	uv run feast -c feature_repo apply
	uv run python scripts/load.py --config $(TRAIN_CONFIG)

split:
	uv run python scripts/split.py --config $(TRAIN_CONFIG)

split-docker:
	$(call RUNNER_WITH_ENV,run --rm runner $(RUNNER_PYTHON) scripts/split.py --config $(TRAIN_CONFIG))

train:
	uv run python scripts/train.py --config $(TRAIN_CONFIG)

load-docker:
	$(call RUNNER_WITH_ENV,run --rm runner $(RUNNER_FEAST) -c feature_repo apply)
	$(call RUNNER_WITH_ENV,run --rm runner $(RUNNER_PYTHON) scripts/load.py --config $(TRAIN_CONFIG))

train-docker:
	$(call RUNNER_WITH_ENV,run --rm runner $(RUNNER_PYTHON) scripts/train.py --config $(TRAIN_CONFIG))

eval:
	uv run python scripts/eval.py

log:
	uv run python scripts/log.py --config $(TRAIN_CONFIG)

pipeline:
	$(call banner,STEP 1: CREATE ENV)
	make setup
	$(call banner,STEP 2: EXTRACT DATA)
	make extract
	$(call banner,STEP 3: PUSH DATA)
	make push
	$(call banner,STEP 4: PULL DATA)
	make pull
	$(call banner,STEP 5: TRANSFORM DATA)
	make transform
	$(call banner,STEP 6: LOAD DATA)
	make load-docker
	$(call banner,STEP 7: SPLIT DATA)
	make split-docker
	$(call banner,STEP 8: TRAIN MODEL)
	make train
	$(call banner,STEP 9: EVALUATE MODEL)
	make eval
	$(call banner,STEP 10: LOG RESULTS)
	make log