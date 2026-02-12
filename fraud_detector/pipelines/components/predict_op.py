"""KFP component -- batch prediction."""

from kfp import dsl

from fraud_detector.pipelines import pipeline_component


@pipeline_component()
def predict_op(
    project_id: str,
    region: str,
    bq_dataset: str,
    feature_table: str,
    predictions_table: str,
    model_display_name: str,
    read_unscored_sql: str,
    scoring_metrics: dsl.Output[dsl.Metrics],
) -> int:
    """Load latest model from registry, score feature data, return count of scored rows."""
    import logging
    import tempfile

    from google.cloud import aiplatform, bigquery, storage

    from fraud_detector import FraudDetector

    logger = logging.getLogger(__name__)

    client = bigquery.Client(project=project_id)
    query = read_unscored_sql.format(
        project_id=project_id,
        bq_dataset=bq_dataset,
        feature_table=feature_table,
        predictions_table=predictions_table,
    )
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        if "Not found" in str(e) and predictions_table in str(e):
            logger.info("Predictions table not found -- scoring all features")
            fallback = f"SELECT * FROM `{project_id}.{bq_dataset}.{feature_table}`"
            df = client.query(fallback).to_dataframe()
        else:
            raise
    logger.info("Loaded %d unscored rows", len(df))

    if df.empty:
        scoring_metrics.log_metric("rows_scored", 0)
        return 0

    # Load model from Vertex AI Model Registry
    aiplatform.init(project=project_id, location=region)
    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="update_time desc",
    )
    if not models:
        raise ValueError(f"No model found with display_name={model_display_name}")

    artifact_uri = models[0].uri
    bucket_name = artifact_uri.replace("gs://", "").split("/")[0]
    blob_prefix = "/".join(artifact_uri.replace("gs://", "").split("/")[1:])
    blob_name = f"{blob_prefix}/model.joblib"

    gcs_client = storage.Client(project=project_id)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        gcs_client.bucket(bucket_name).blob(blob_name).download_to_filename(f.name)
        fd = FraudDetector()
        fd.load_model(f.name)

    df = fd.predict(df)
    output_cols = [
        "tx_id",
        "tx_ts",
        "customer_id",
        "terminal_id",
        "tx_amount",
        "fraud_probability",
        "fraud_prediction",
        "scored_at",
    ]
    table_ref = f"{project_id}.{bq_dataset}.{predictions_table}"
    from google.cloud.bigquery import LoadJobConfig

    job_config = LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df[output_cols], table_ref, job_config=job_config)
    job.result()
    logger.info("Predictions written to %s", table_ref)

    # Log scoring stats
    scoring_metrics.log_metric("rows_scored", len(df))
    scoring_metrics.log_metric("avg_fraud_probability", float(df["fraud_probability"].mean()))
    scoring_metrics.log_metric("fraud_rate_predicted", float(df["fraud_prediction"].mean()))

    return len(df)
