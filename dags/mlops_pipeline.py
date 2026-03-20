from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


REPO_HOST_DIR = os.environ.get("MLOPS_EXAMPLES_HOST_DIR")
if not REPO_HOST_DIR:
    raise RuntimeError(
        "Set MLOPS_EXAMPLES_HOST_DIR to the host path of your mlops-examples checkout."
    )
REPO_CONTAINER_DIR = "/work"
RUNNER_IMAGE = "mlops-examples-runner"
NETWORK_NAME = os.environ.get("MLOPS_NETWORK", "mlops")

COMMON_ENV = {
    "PUBLIC_FQDN": os.environ.get("PUBLIC_FQDN", "localhost"),
    "MLFLOW_BASE_PATH": os.environ.get("MLFLOW_BASE_PATH", "mlflow"),
    # log.py prefers MLFLOW_TRACKING_URI from the environment, so use the
    # internal Docker-network address here instead of the external nginx path.
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", "postgres"),
    "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5432"),
    "POSTGRES_USER": os.environ.get("POSTGRES_USER", ""),
    "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    "MLFLOW_TRACKING_USERNAME": (
        os.environ.get("MLFLOW_TRACKING_USERNAME")
        or os.environ.get("MLFLOW_AUTH_ADMIN_USERNAME", "")
    ),
    "MLFLOW_TRACKING_PASSWORD": (
        os.environ.get("MLFLOW_TRACKING_PASSWORD")
        or os.environ.get("MLFLOW_AUTH_ADMIN_PASSWORD", "")
    ),
    "PYTHONPATH": f"{REPO_CONTAINER_DIR}/src",
}

COMMON_MOUNTS = [
    Mount(source=REPO_HOST_DIR, target=REPO_CONTAINER_DIR, type="bind"),
]


def runner_task(task_id: str, command: str, *, skip_on_exit_code: int | None = None) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=RUNNER_IMAGE,
        command=["/bin/bash", "-lc", command],
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=COMMON_MOUNTS,
        working_dir=REPO_CONTAINER_DIR,
        network_mode=NETWORK_NAME,
        environment=COMMON_ENV,
        skip_on_exit_code=skip_on_exit_code,
    )


with DAG(
    dag_id="mlops_pipeline",
    description="Runs the MLOps example pipeline as one Airflow task per stage.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mlops", "pipeline"],
) as dag:
    setup_environment = BashOperator(
        task_id="setup_environment",
        bash_command=(
            "if docker image inspect "
            f"{RUNNER_IMAGE} "
            "> /dev/null 2>&1; then "
            f"echo '{RUNNER_IMAGE} already exists; skipping rebuild.'; "
            "else "
            "docker build "
            f"-t {RUNNER_IMAGE} "
            "-f /opt/mlops-examples/Dockerfile.runner "
            "/opt/mlops-examples; "
            "fi"
        ),
    )

    extract_data = runner_task(
        "extract_data",
        (
            "/opt/venv/bin/python -m mlops_examples.cli.extract --out data/raw/breast_cancer.csv && "
            "/opt/venv/bin/dvc add data/raw/breast_cancer.csv"
        ),
    )

    push_data = runner_task(
        "push_data",
        (
            "trap 'rm -f .dvc/config.local' EXIT; "
            "/opt/venv/bin/dvc remote modify --local rustfs endpointurl "
            "http://rustfs:9000 >/dev/null; "
            "/opt/venv/bin/dvc push"
        ),
    )

    pull_data = runner_task(
        "pull_data",
        (
            "trap 'rm -f .dvc/config.local' EXIT; "
            "/opt/venv/bin/dvc remote modify --local rustfs endpointurl "
            "http://rustfs:9000 >/dev/null; "
            "/opt/venv/bin/dvc pull"
        ),
    )

    transform_data = runner_task(
        "transform_data",
        "/opt/venv/bin/python -m mlops_examples.cli.transform",
    )

    snapshot_features = runner_task(
        "snapshot_features",
        (
            "/opt/venv/bin/python -m mlops_examples.cli.snapshot --config configs/dev.yaml && "
            "/opt/venv/bin/dvc add data/features/current"
        ),
    )

    push_snapshot = runner_task(
        "push_snapshot",
        (
            "trap 'rm -f .dvc/config.local' EXIT; "
            "/opt/venv/bin/dvc remote modify --local rustfs endpointurl "
            "http://rustfs:9000 >/dev/null; "
            "/opt/venv/bin/dvc push"
        ),
    )

    split_data = runner_task(
        "split_data",
        (
            "/opt/venv/bin/feast -c feature_store apply && "
            "/opt/venv/bin/python -m mlops_examples.cli.split --config configs/dev.yaml"
        ),
    )

    train_model = runner_task(
        "train_model",
        "/opt/venv/bin/python -m mlops_examples.cli.train --config configs/dev.yaml",
    )

    evaluate_model = runner_task(
        "evaluate_model",
        "/opt/venv/bin/python -m mlops_examples.cli.eval --config configs/dev.yaml",
    )

    log_to_mlflow = runner_task(
        "log_to_mlflow",
        (
            "if [ -f .env.user ]; then set -a; . ./.env.user; set +a; fi; "
            "/opt/venv/bin/python -m mlops_examples.cli.log --config configs/dev.yaml"
        ),
    )

    (
        setup_environment
        >> extract_data
        >> push_data
        >> pull_data
        >> transform_data
        >> snapshot_features
        >> push_snapshot
        >> split_data
        >> train_model
        >> evaluate_model
        >> log_to_mlflow
    )
