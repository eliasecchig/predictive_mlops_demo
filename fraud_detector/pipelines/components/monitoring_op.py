"""KFP component -- Vertex AI Model Monitoring v2 setup."""

from fraud_detector.pipelines import pipeline_component


@pipeline_component()
def setup_monitoring_op(
    project_id: str,
    region: str,
    bq_dataset: str,
    feature_table: str,
    predictions_table: str,
    model_resource_name: str,
    alert_emails: str,
    default_drift_threshold: float = 0.3,
    monitoring_schedule: str = "0 8 * * 1",
) -> str:
    """Set up Vertex AI Model Monitoring v2 for a registered model.

    Configures feature drift detection (Jensen-Shannon divergence) comparing
    the training features table against the predictions/scores table.

    Args:
        project_id: GCP project ID.
        region: GCP region.
        bq_dataset: BigQuery dataset name.
        feature_table: BigQuery table with training features (baseline).
        predictions_table: BigQuery table with prediction outputs (target).
        model_resource_name: Full Vertex AI model resource name from register_op.
        alert_emails: Comma-separated email addresses for drift alerts.
        default_drift_threshold: Default Jensen-Shannon divergence threshold.
        monitoring_schedule: Cron expression for monitoring schedule.

    Returns:
        Monitor resource name, or skip/error status string.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("-" * 60)
    logger.info("[MON] STEP: Model Monitoring Setup")
    logger.info("-" * 60)

    # Skip if model was not registered
    if model_resource_name in ("NOT_REGISTERED", "LOCAL_ONLY"):
        logger.info("[SKIP] Skipping monitoring setup -- model status: %s", model_resource_name)
        return f"SKIPPED:{model_resource_name}"

    try:
        from google.cloud import aiplatform
        from vertexai.resources.preview.ml_monitoring import ModelMonitor
        from vertexai.resources.preview.ml_monitoring.spec import (
            DataDriftSpec,
            FieldSchema,
            ModelMonitoringSchema,
            MonitoringInput,
            NotificationSpec,
            OutputSpec,
            TabularObjective,
        )

        from fraud_detector import FraudDetector

        aiplatform.init(project=project_id, location=region)

        # Build schema: all feature columns are float, plus prediction field
        feature_cols = FraudDetector.feature_columns()
        feature_schemas = [FieldSchema(name=col, data_type="float") for col in feature_cols]
        feature_schemas.append(FieldSchema(name="fraud_probability", data_type="float"))

        schema = ModelMonitoringSchema(
            feature_fields=feature_schemas,
            prediction_fields=[FieldSchema(name="fraud_prediction", data_type="integer")],
        )

        # Extract model ID and version from resource name
        # register_op returns: projects/.../models/<model_id> (unversioned)
        # or projects/.../models/<model_id>@<version>
        model_suffix = model_resource_name.split("/models/")[-1]
        if "@" in model_suffix:
            model_id, model_version = model_suffix.split("@", 1)
        else:
            model_id = model_suffix.split("/")[0]
            model_version = "1"
        monitor_display_name = f"fraud-detector-monitor-{model_id}"

        # Clean up any existing monitor with the same display name
        existing_monitors = ModelMonitor.list(
            filter=f'display_name="{monitor_display_name}"',
        )
        for old_monitor in existing_monitors:
            logger.info("[DEL] Deleting existing monitor: %s", old_monitor.name)
            old_monitor.delete(force=True)

        # Create model monitor
        monitor = ModelMonitor.create(
            project=project_id,
            location=region,
            display_name=monitor_display_name,
            model_name=model_resource_name,
            model_version_id=model_version,
            model_monitoring_schema=schema,
        )
        logger.info("[OK] Model monitor created: %s", monitor.name)

        # BigQuery data sources
        baseline_uri = f"bq://{project_id}.{bq_dataset}.{feature_table}"
        target_uri = f"bq://{project_id}.{bq_dataset}.{predictions_table}"

        # Parse email list
        emails = [e.strip() for e in alert_emails.split(",") if e.strip()]

        # Create scheduled monitoring run
        monitor.create_schedule(
            display_name=f"{monitor_display_name}-schedule",
            cron=monitoring_schedule,
            baseline_dataset=MonitoringInput(table_uri=baseline_uri),
            target_dataset=MonitoringInput(table_uri=target_uri),
            tabular_objective_spec=TabularObjective(
                feature_drift_spec=DataDriftSpec(
                    default_numeric_alert_threshold=default_drift_threshold,
                    default_categorical_alert_threshold=default_drift_threshold,
                ),
            ),
            notification_spec=NotificationSpec(user_emails=emails),
            output_spec=OutputSpec(
                gcs_base_dir=f"gs://{project_id}-fraud-detector-pipeline-root/monitoring",
            ),
        )
        logger.info("[SCHED] Monitoring schedule created for %s", monitor.name)

        return monitor.name

    except Exception:
        logger.exception("[ERR] Monitoring setup failed (non-blocking)")
        return "ERROR:monitoring_setup_failed"
