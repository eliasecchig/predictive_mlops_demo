"""Config loader -- reads YAML config files and resolves environment variables."""

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"
SQL_DIR = CONFIG_DIR / "sql"


def load_config(config_name: str) -> dict:
    """Load a YAML config file from the config/ directory.

    Environment variable placeholders like ${PROJECT_ID} are resolved
    from the environment. If PROJECT_ID is not set, it is auto-detected
    from the active gcloud configuration.
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    text = config_path.read_text()
    # Auto-resolve PROJECT_ID if not in environment
    if "${PROJECT_ID}" in text and "PROJECT_ID" not in os.environ:
        os.environ["PROJECT_ID"] = get_project_id()
    # Resolve ${VAR} placeholders
    for key, value in os.environ.items():
        text = text.replace(f"${{{key}}}", value)
    return yaml.safe_load(text)


def load_sql(name: str) -> str:
    """Load a SQL template from the config/sql/ directory.

    Returns the raw template string with {placeholder} variables
    ready for ``.format()`` substitution.
    """
    return (SQL_DIR / name).read_text()


def get_project_id() -> str:
    """Return the active GCP project ID from the environment or gcloud."""
    project_id = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id
    # Fallback: read from gcloud config
    import subprocess

    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
