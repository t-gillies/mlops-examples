.PHONY: help setup lock test test-docker data data-append pull push pull-host push-host pull-docker push-docker runner-build snapshot snapshot-docker load load-docker split split-docker train train-docker eval log pipeline

help:
	@echo "Make targets:"
	@echo "  make setup         Create virtual environment and install dependencies"
	@echo "  make lock          Refresh uv.lock"
	@echo "  make test          Run the root unit test suite locally"
	@echo "  make test-docker   Run the root unit test suite inside the runner image"
	@echo "  make extract          Generate deterministic dataset and track with DVC"
	@echo "  make extract-append   Append one row to dataset and track with DVC"
	@echo "  make pull          Pull DVC data inside docker using the internal RustFS service"
	@echo "  make push          Push DVC data inside docker using the internal RustFS service"
	@echo "  make pull-host     Host-based DVC pull if a direct S3 endpoint is reachable"
	@echo "  make push-host     Host-based DVC push if a direct S3 endpoint is reachable"
	@echo "  make pull-docker   Alias for make pull"
	@echo "  make push-docker   Alias for make push"
	@echo "  make runner-build  Build/update the docker runner image"
	@echo "  make transform      Transform raw data into processed data"
	@echo "  make snapshot       Build a local Parquet feature snapshot and track it with DVC"
	@echo "  make snapshot-docker  Build a Parquet feature snapshot inside docker and track it with DVC"
	@echo "  make load           Alias for make snapshot"
	@echo "  make split      Split data into train/val/test sets"
	@echo "  make train         Host-based training workflow"
	@echo "  make load-docker  Alias for make snapshot-docker"
	@echo "  make train-docker     Train inside docker on the shared mlops network"
	@echo "  make eval          Run model evaluation and locally create metrics/plots artifacts"
	@echo "  make log           Log training metrics to MLFlow"
	@echo "  make pipeline          Run the full pipeline end-to-end (except for setup and lock)"

	@echo "  Variables: TRAIN_CONFIG=configs/dev.yaml MLOPS_SERVICES_DIR=../mlops-services"

MLOPS_SERVICES_DIR ?= ../mlops-services
TRAIN_CONFIG ?= configs/dev.yaml
RUNNER_INTERNAL_POSTGRES_HOST = postgres
RUNNER_INTERNAL_MLFLOW_URI = http://mlflow:5000
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
	export MLFLOW_TRACKING_USERNAME="$${MLFLOW_TRACKING_USERNAME:-$${MLFLOW_AUTH_ADMIN_USERNAME:-}}"; \
	export MLFLOW_TRACKING_PASSWORD="$${MLFLOW_TRACKING_PASSWORD:-$${MLFLOW_AUTH_ADMIN_PASSWORD:-}}"; \
	case "$${POSTGRES_HOST:-}" in \
	  "" ) export POSTGRES_HOST="$(RUNNER_INTERNAL_POSTGRES_HOST)" ;; \
	  127.0.0.1|localhost|0.0.0.0 ) export POSTGRES_HOST="$(RUNNER_INTERNAL_POSTGRES_HOST)" ;; \
	esac; \
	case "$${MLFLOW_TRACKING_URI:-}" in \
	  "" ) export MLFLOW_TRACKING_URI="$(RUNNER_INTERNAL_MLFLOW_URI)" ;; \
	  http://127.0.0.1|http://localhost|http://0.0.0.0 ) export MLFLOW_TRACKING_URI="$(RUNNER_INTERNAL_MLFLOW_URI)" ;; \
	  http://127.0.0.1/*|http://localhost/*|http://0.0.0.0/* ) export MLFLOW_TRACKING_URI="$(RUNNER_INTERNAL_MLFLOW_URI)" ;; \
	esac; \
	: "$${POSTGRES_USER:?Set POSTGRES_USER in your shell or mlops-services env}"; \
	: "$${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in your shell or mlops-services env}"; \
	: "$${MLFLOW_TRACKING_USERNAME:?Set MLFLOW_TRACKING_USERNAME in your shell or .env.user}"; \
	: "$${MLFLOW_TRACKING_PASSWORD:?Set MLFLOW_TRACKING_PASSWORD in your shell or .env.user}"; \
	$(RUNNER_COMPOSE) $(1)'
endef

define RUNNER_BUILD_WITH_ENV
	/bin/bash -lc 'set -euo pipefail; \
	CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	USER_ENV=".env.user"; \
	if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	if [ -f "$$USER_ENV" ]; then set -a; source "$$USER_ENV"; set +a; fi; \
	$(RUNNER_COMPOSE) build runner'
endef

define RUNNER_TEST_WITH_ENV
	/bin/bash -lc 'set -euo pipefail; \
	CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	USER_ENV=".env.user"; \
	if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	if [ -f "$$USER_ENV" ]; then set -a; source "$$USER_ENV"; set +a; fi; \
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mlops-examples-tests-mpl $(RUNNER_COMPOSE) run --rm runner $(RUNNER_PYTHON) -m unittest discover -s tests -v'
endef

define RUNNER_WITH_POSTGRES_ENV
	/bin/bash -lc 'set -euo pipefail; \
	CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	USER_ENV=".env.user"; \
	if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	if [ -f "$$USER_ENV" ]; then set -a; source "$$USER_ENV"; set +a; fi; \
	export MLFLOW_TRACKING_USERNAME="$${MLFLOW_TRACKING_USERNAME:-$${MLFLOW_AUTH_ADMIN_USERNAME:-}}"; \
	export MLFLOW_TRACKING_PASSWORD="$${MLFLOW_TRACKING_PASSWORD:-$${MLFLOW_AUTH_ADMIN_PASSWORD:-}}"; \
	case "$${POSTGRES_HOST:-}" in \
	  "" ) export POSTGRES_HOST="$(RUNNER_INTERNAL_POSTGRES_HOST)" ;; \
	  127.0.0.1|localhost|0.0.0.0 ) export POSTGRES_HOST="$(RUNNER_INTERNAL_POSTGRES_HOST)" ;; \
	esac; \
	case "$${MLFLOW_TRACKING_URI:-}" in \
	  "" ) export MLFLOW_TRACKING_URI="$(RUNNER_INTERNAL_MLFLOW_URI)" ;; \
	  http://127.0.0.1|http://localhost|http://0.0.0.0 ) export MLFLOW_TRACKING_URI="$(RUNNER_INTERNAL_MLFLOW_URI)" ;; \
	  http://127.0.0.1/*|http://localhost/*|http://0.0.0.0/* ) export MLFLOW_TRACKING_URI="$(RUNNER_INTERNAL_MLFLOW_URI)" ;; \
	esac; \
	: "$${POSTGRES_USER:?Set POSTGRES_USER in your shell or mlops-services env}"; \
	: "$${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in your shell or mlops-services env}"; \
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
	export AWS_ACCESS_KEY_ID="$${AWS_ACCESS_KEY_ID:-$${RUSTFS_ACCESS_KEY:-}}"; \
	export AWS_SECRET_ACCESS_KEY="$${AWS_SECRET_ACCESS_KEY:-$${RUSTFS_SECRET_KEY:-}}"; \
	: "$${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID or provide RUSTFS_ACCESS_KEY via mlops-services env}"; \
	: "$${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY or provide RUSTFS_SECRET_KEY via mlops-services env}"; \
	$(RUNNER_COMPOSE) run --rm runner sh -lc '"'"'trap "rm -f .dvc/config.local" EXIT; $(RUNNER_DVC) remote modify --local rustfs endpointurl http://rustfs:9000 >/dev/null; $(RUNNER_DVC) $(1)'"'"''
endef

define LOG_WITH_ENV
	/bin/bash -lc 'set -euo pipefail; \
	CONFIG_ENV="$(MLOPS_SERVICES_DIR)/env/config.env"; \
	SECRETS_ENV="$(MLOPS_SERVICES_DIR)/env/secrets.env"; \
	USER_ENV=".env.user"; \
	if [ -f "$$CONFIG_ENV" ]; then set -a; source "$$CONFIG_ENV"; set +a; fi; \
	if [ -f "$$SECRETS_ENV" ]; then set -a; source "$$SECRETS_ENV"; set +a; fi; \
	if [ -f "$$USER_ENV" ]; then set -a; source "$$USER_ENV"; set +a; fi; \
	export MLFLOW_TRACKING_USERNAME="$${MLFLOW_TRACKING_USERNAME:-$${MLFLOW_AUTH_ADMIN_USERNAME:-}}"; \
	export MLFLOW_TRACKING_PASSWORD="$${MLFLOW_TRACKING_PASSWORD:-$${MLFLOW_AUTH_ADMIN_PASSWORD:-}}"; \
	export PYTHONPATH="src$${PYTHONPATH:+:$$PYTHONPATH}"; \
	: "$${MLFLOW_TRACKING_USERNAME:?Set MLFLOW_TRACKING_USERNAME in your shell or .env.user}"; \
	: "$${MLFLOW_TRACKING_PASSWORD:?Set MLFLOW_TRACKING_PASSWORD in your shell or .env.user}"; \
	uv run python -m mlops_examples.cli.log --config $(TRAIN_CONFIG)'
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

test:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mlops-examples-tests-mpl PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

test-docker:
	$(call RUNNER_TEST_WITH_ENV)

extract:
	PYTHONPATH=src uv run python -m mlops_examples.cli.extract --out data/raw/breast_cancer.csv
	uv run dvc add data/raw/breast_cancer.csv

extract-append:
	PYTHONPATH=src uv run python -m mlops_examples.cli.extract --append-row --out data/raw/breast_cancer.csv
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
	$(call RUNNER_BUILD_WITH_ENV)

transform:
	PYTHONPATH=src uv run python -m mlops_examples.cli.transform

snapshot:
	PYTHONPATH=src uv run python -m mlops_examples.cli.snapshot --config $(TRAIN_CONFIG)
	uv run dvc add data/features/current

load: snapshot

split:
	uv run feast -c feature_store apply
	PYTHONPATH=src uv run python -m mlops_examples.cli.split --config $(TRAIN_CONFIG)

split-docker:
	$(call RUNNER_WITH_POSTGRES_ENV,run --rm runner $(RUNNER_FEAST) -c feature_store apply)
	$(call RUNNER_WITH_POSTGRES_ENV,run --rm runner $(RUNNER_PYTHON) -m mlops_examples.cli.split --config $(TRAIN_CONFIG))

train:
	PYTHONPATH=src uv run python -m mlops_examples.cli.train --config $(TRAIN_CONFIG)

snapshot-docker:
	$(call RUNNER_WITH_POSTGRES_ENV,run --rm runner $(RUNNER_PYTHON) -m mlops_examples.cli.snapshot --config $(TRAIN_CONFIG))
	$(call RUNNER_WITH_POSTGRES_ENV,run --rm runner $(RUNNER_DVC) add data/features/current)

load-docker: snapshot-docker

train-docker:
	$(call RUNNER_WITH_POSTGRES_ENV,run --rm runner $(RUNNER_PYTHON) -m mlops_examples.cli.train --config $(TRAIN_CONFIG))

eval:
	PYTHONPATH=src uv run python -m mlops_examples.cli.eval --config $(TRAIN_CONFIG)

log:
	$(call LOG_WITH_ENV)

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
	$(call banner,STEP 6: BUILD FEATURE SNAPSHOT)
	make snapshot-docker
	$(call banner,STEP 7: PUSH SNAPSHOT)
	make push
	$(call banner,STEP 8: SPLIT DATA)
	make split-docker
	$(call banner,STEP 9: TRAIN MODEL)
	make train
	$(call banner,STEP 10: EVALUATE MODEL)
	make eval
	$(call banner,STEP 11: LOG RESULTS)
	make log
