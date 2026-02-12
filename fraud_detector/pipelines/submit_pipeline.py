"""Compile and submit KFP pipelines to Vertex AI (or run locally)."""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

from fraud_detector.config import load_config, load_sql

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = PROJECT_ROOT / "fraud_detector" / "_version.py"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _compute_deps_hash() -> str:
    """Compute a content-based hash of dependency files for image tagging.

    Hashes Dockerfile, pyproject.toml, and uv.lock only.  The base image
    only needs to be rebuilt when dependencies change.
    """
    h = hashlib.sha256()
    for name in ["Dockerfile", "pyproject.toml", "uv.lock"]:
        path = PROJECT_ROOT / name
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _compute_code_hash() -> str:
    """Compute a content-based hash of Python source files.

    Used for the wheel version suffix so each code change gets a unique
    version in Artifact Registry.
    """
    h = hashlib.sha256()
    for path in sorted(PROJECT_ROOT.glob("fraud_detector/**/*.py")):
        h.update(str(path.relative_to(PROJECT_ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Container image management (deps-only)
# ---------------------------------------------------------------------------


def _get_image_uri(tag: str) -> str:
    """Build the full Artifact Registry image URI."""
    region = os.environ.get("REGION", "us-central1")
    cicd_project = os.environ.get("CICD_PROJECT_ID") or os.environ.get("PROJECT_ID", "")
    return f"{region}-docker.pkg.dev/{cicd_project}/fraud-detector-docker/fraud-detector:{tag}"


def _image_exists(image_uri: str) -> bool:
    """Check if a Docker image tag already exists in Artifact Registry."""
    result = subprocess.run(
        ["gcloud", "artifacts", "docker", "images", "describe", image_uri],
        capture_output=True,
    )
    return result.returncode == 0


def _docker_available() -> bool:
    """Check if the local Docker daemon is running."""
    result = subprocess.run(["docker", "info"], capture_output=True)
    return result.returncode == 0


def _build_and_push(image_uri: str) -> None:
    """Build and push the pipeline container image.

    Prefers local Docker (faster) and falls back to Cloud Build.
    Always targets linux/amd64 since Vertex AI runs on AMD64.
    """
    registry = image_uri.split("/")[0]
    cicd_project = os.environ.get("CICD_PROJECT_ID") or os.environ.get("PROJECT_ID", "")

    if _docker_available():
        print(f"Building with Docker: {image_uri}")
        subprocess.run(
            ["gcloud", "auth", "configure-docker", registry, "--quiet"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "buildx", "build", "--platform", "linux/amd64", "-t", image_uri, "--push", str(PROJECT_ROOT)],
            check=True,
        )
    else:
        print(f"Building with Cloud Build: {image_uri}")
        subprocess.run(
            ["gcloud", "builds", "submit", "--tag", image_uri, "--project", cicd_project, "--quiet", str(PROJECT_ROOT)],
            check=True,
        )

    print(f"Image ready: {image_uri}")


def ensure_deps_image() -> None:
    """Ensure the deps-only container image is built and pushed.

    - If IMAGE_TAG is already set (CI/CD), uses it as-is -- no build.
    - Otherwise computes a deps hash, checks Artifact Registry,
      and builds only if the image doesn't exist yet.
    """
    if os.environ.get("IMAGE_TAG"):
        return  # CI/CD already built and tagged the image

    tag = _compute_deps_hash()
    image_uri = _get_image_uri(tag)

    if _image_exists(image_uri):
        print(f"Image up to date: {image_uri}")
    else:
        _build_and_push(image_uri)

    # Set for BASE_IMAGE resolution when pipeline modules are imported
    os.environ["IMAGE_TAG"] = tag


# ---------------------------------------------------------------------------
# Code package management (wheel → Artifact Registry Python repo)
# ---------------------------------------------------------------------------


def _get_ar_repo_url() -> str:
    """Return the AR Python repo upload URL."""
    region = os.environ.get("REGION", "us-central1")
    project = os.environ.get("CICD_PROJECT_ID") or os.environ.get("PROJECT_ID", "")
    return f"https://{region}-python.pkg.dev/{project}/fraud-detector-python/"


def _wheel_exists(version: str) -> bool:
    """Check if a wheel version already exists in Artifact Registry."""
    region = os.environ.get("REGION", "us-central1")
    project = os.environ.get("CICD_PROJECT_ID") or os.environ.get("PROJECT_ID", "")
    # Use only the hash suffix in the filter to avoid regex issues with '+'
    hash_suffix = version.split("+")[-1] if "+" in version else version
    result = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "versions",
            "list",
            "--repository=fraud-detector-python",
            f"--location={region}",
            f"--project={project}",
            "--package=fraud-detector",
            f"--filter=name~{hash_suffix}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and version in result.stdout


def ensure_code_package() -> None:
    """Build and upload the fraud-detector wheel to Artifact Registry.

    - Computes a hash of all Python source files.
    - If CODE_VERSION is already set (CI/CD), uses it as-is.
    - Otherwise checks if the version exists in AR; if not, builds
      and uploads the wheel.
    - Temporarily patches ``_version.py`` with ``0.1.0+{hash}``
      during the build, then restores the original.
    """
    if os.environ.get("CODE_VERSION"):
        return  # CI/CD already built and uploaded

    code_hash = _compute_code_hash()
    version = f"0.1.0+{code_hash}"

    if _wheel_exists(version):
        print(f"Code package up to date: fraud-detector=={version}")
    else:
        # Temporarily patch _version.py
        original = VERSION_FILE.read_text()
        try:
            VERSION_FILE.write_text(f'__version__ = "{version}"\n')

            # Build wheel
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(PROJECT_ROOT / "dist")],
                check=True,
                cwd=str(PROJECT_ROOT),
            )

            # Upload to AR using gcloud credentials
            upload_url = _get_ar_repo_url()
            token = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            subprocess.run(
                [
                    "uv",
                    "publish",
                    "--publish-url",
                    upload_url,
                    "--username",
                    "oauth2accesstoken",
                    "--password",
                    token,
                    str(PROJECT_ROOT / "dist" / f"fraud_detector-{version}-py3-none-any.whl"),
                ],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            print(f"Code package uploaded: fraud-detector=={version}")
        finally:
            VERSION_FILE.write_text(original)

    os.environ["CODE_VERSION"] = version


# ---------------------------------------------------------------------------
# Pipeline compilation and execution
# ---------------------------------------------------------------------------


def _resolve_sql(config: dict) -> dict[str, str]:
    """Load all SQL templates referenced in the config's ``sql`` block."""
    return {key: load_sql(filename) for key, filename in config.get("sql", {}).items()}


def _enable_caching(config: dict) -> bool:
    """Return caching flag: enabled by default, disabled for prod."""
    env = os.environ.get("ENVIRONMENT", "dev").lower()
    if env == "prod":
        return False
    return config.get("enable_caching", True)


def compile_pipeline(pipeline_name: str) -> str:
    """Compile a pipeline and return the path to the compiled JSON."""
    from kfp import compiler

    if pipeline_name == "training":
        from fraud_detector.pipelines.training_pipeline import training_pipeline

        pipeline_func = training_pipeline
    elif pipeline_name == "scoring":
        from fraud_detector.pipelines.scoring_pipeline import scoring_pipeline

        pipeline_func = scoring_pipeline
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    output_path = f"{pipeline_name}_pipeline.json"
    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=output_path)
    print(f"Pipeline compiled: {output_path}")
    return output_path


def run_local(pipeline_name: str, config: dict) -> None:
    """Run a pipeline locally using KFP local runner."""
    from kfp import local

    local.init(runner=local.SubprocessRunner(use_venv=False))

    sql = _resolve_sql(config)

    if pipeline_name == "training":
        from fraud_detector.pipelines.training_pipeline import training_pipeline

        monitoring_config = load_config("monitoring")
        xgb_params = config.get("xgb_params", {})
        training_pipeline(
            project_id=config["project_id"],
            region=config["region"],
            bq_dataset=config["bq_dataset"],
            feature_table=config.get("feature_table", "fraud_features"),
            model_display_name=config["model_display_name"],
            split_date=config.get("train_test_split_date", "2023-06-01"),
            threshold_auc=config.get("eval_threshold_auc", 0.85),
            read_raw_sql=sql["read_raw"],
            read_features_sql=sql["read_features"],
            max_depth=xgb_params.get("max_depth", 6),
            n_estimators=xgb_params.get("n_estimators", 200),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            scale_pos_weight=float(xgb_params.get("scale_pos_weight", 10.0)),
            alert_emails=",".join(monitoring_config.get("alert_emails", [])),
            default_drift_threshold=float(min(monitoring_config.get("drift_thresholds", {}).values() or [0.3])),
            predictions_table=monitoring_config.get("predictions_table", "fraud_scores"),
            monitoring_schedule=monitoring_config.get("schedule", "0 8 * * 1"),
            skip_profiling=True,
        )
    elif pipeline_name == "scoring":
        from fraud_detector.pipelines.scoring_pipeline import scoring_pipeline

        scoring_pipeline(
            project_id=config["project_id"],
            region=config["region"],
            bq_dataset=config["bq_dataset"],
            feature_table=config.get("feature_table", "fraud_features"),
            model_display_name=config["model_display_name"],
            predictions_table=config.get("predictions_table", "fraud_scores"),
            read_raw_sql=sql["read_raw"],
            read_unscored_sql=sql["read_unscored"],
        )


def submit_to_vertex(
    pipeline_name: str,
    config: dict,
    schedule_only: bool = False,
    cron_schedule: str | None = None,
    experiment: str | None = None,
) -> None:
    """Submit a compiled pipeline to Vertex AI."""
    from google.cloud import aiplatform

    project_id = config["project_id"]
    region = config["region"]
    experiment_name = experiment or f"fraud-{pipeline_name}-experiment"
    aiplatform.init(project=project_id, location=region, experiment=experiment_name)

    compiled_path = compile_pipeline(pipeline_name)
    sql = _resolve_sql(config)
    caching = _enable_caching(config)

    # Build pipeline params from config
    if pipeline_name == "training":
        monitoring_config = load_config("monitoring")
        xgb_params = config.get("xgb_params", {})
        params = {
            "project_id": project_id,
            "region": region,
            "bq_dataset": config["bq_dataset"],
            "feature_table": config.get("feature_table", "fraud_features"),
            "model_display_name": config["model_display_name"],
            "split_date": config.get("train_test_split_date", "2023-06-01"),
            "threshold_auc": config.get("eval_threshold_auc", 0.85),
            "read_raw_sql": sql["read_raw"],
            "read_features_sql": sql["read_features"],
            "max_depth": xgb_params.get("max_depth", 6),
            "n_estimators": xgb_params.get("n_estimators", 200),
            "learning_rate": xgb_params.get("learning_rate", 0.1),
            "scale_pos_weight": xgb_params.get("scale_pos_weight", 10.0),
            "alert_emails": ",".join(monitoring_config.get("alert_emails", [])),
            "default_drift_threshold": float(min(monitoring_config.get("drift_thresholds", {}).values() or [0.3])),
            "predictions_table": monitoring_config.get("predictions_table", "fraud_scores"),
            "monitoring_schedule": monitoring_config.get("schedule", "0 8 * * 1"),
        }
    else:
        params = {
            "project_id": project_id,
            "region": region,
            "bq_dataset": config["bq_dataset"],
            "feature_table": config.get("feature_table", "fraud_features"),
            "model_display_name": config["model_display_name"],
            "predictions_table": config.get("predictions_table", "fraud_scores"),
            "read_raw_sql": sql["read_raw"],
            "read_unscored_sql": sql["read_unscored"],
        }

    pipeline_root = f"gs://{project_id}-fraud-detector-pipeline-root"
    display_name = config.get("pipeline_name", f"fraud-detector-{pipeline_name}")
    pipeline_sa = os.environ.get("PIPELINE_SA_EMAIL")

    if schedule_only:
        schedule = cron_schedule or config.get("schedule", "0 2 * * 0")

        job = aiplatform.PipelineJob(
            display_name=display_name,
            template_path=compiled_path,
            pipeline_root=pipeline_root,
            parameter_values=params,
            enable_caching=caching,
        )

        job.create_schedule(
            display_name=f"{display_name}-schedule",
            cron=schedule,
            service_account=pipeline_sa,
        )
        print(f"Schedule created: {display_name} -- cron: {schedule}")
    else:
        job = aiplatform.PipelineJob(
            display_name=display_name,
            template_path=compiled_path,
            pipeline_root=pipeline_root,
            parameter_values=params,
            enable_caching=caching,
        )
        job.submit(service_account=pipeline_sa, experiment=experiment_name)
        print(f"Pipeline submitted: {display_name} (experiment: {experiment_name})")


def main():
    parser = argparse.ArgumentParser(description="Submit KFP pipelines")
    parser.add_argument("--pipeline", required=True, choices=["training", "scoring"], help="Pipeline to run")
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to Vertex AI")
    parser.add_argument("--schedule-only", action="store_true", help="Create/update schedule without running")
    parser.add_argument("--cron-schedule", type=str, help="Override cron schedule")
    parser.add_argument("--compile-only", action="store_true", help="Compile pipeline without submitting")
    parser.add_argument("--experiment", type=str, help="Vertex AI experiment name (default: fraud-detector-{pipeline})")
    args = parser.parse_args()

    config = load_config(args.pipeline)

    if args.compile_only:
        # Set IMAGE_TAG and CODE_VERSION from hashes for correct URIs in compiled output
        if not os.environ.get("IMAGE_TAG"):
            os.environ["IMAGE_TAG"] = _compute_deps_hash()
        if not os.environ.get("CODE_VERSION"):
            os.environ["CODE_VERSION"] = f"0.1.0+{_compute_code_hash()}"
        compile_pipeline(args.pipeline)
    elif args.local:
        run_local(args.pipeline, config)
    else:
        ensure_deps_image()  # Build + push deps image if needed, sets IMAGE_TAG
        ensure_code_package()  # Build + upload wheel if needed, sets CODE_VERSION
        submit_to_vertex(
            args.pipeline,
            config,
            schedule_only=args.schedule_only,
            cron_schedule=args.cron_schedule,
            experiment=args.experiment,
        )


if __name__ == "__main__":
    main()
