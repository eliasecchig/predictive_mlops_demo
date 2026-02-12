"""KFP component -- feature engineering."""

from fraud_detector.pipelines import pipeline_component


@pipeline_component()
def feature_engineering_op(
    project_id: str,
    bq_dataset: str,
    feature_table: str,
    read_raw_sql: str,
) -> str:
    """Read raw data from BQ, compute rolling features, write feature table back to BQ."""
    import logging

    import pandas as pd
    from google.cloud import bigquery

    from fraud_detector import FraudDetector

    logger = logging.getLogger(__name__)

    logger.info("-" * 60)
    logger.info("[FE] STEP: Feature Engineering")
    logger.info("-" * 60)

    client = bigquery.Client(project=project_id)

    logger.info("[IN] Reading raw data from BigQuery…")
    query = read_raw_sql.format(project_id=project_id, bq_dataset=bq_dataset)
    df = client.query(query).to_dataframe()
    df["tx_ts"] = pd.to_datetime(df["tx_ts"])
    logger.info("[DATA] Loaded %d rows from BigQuery", len(df))

    logger.info("[PROC] Computing rolling-window features…")
    df = FraudDetector.compute_features(df)

    table_ref = f"{project_id}.{bq_dataset}.{feature_table}"
    from google.cloud.bigquery import LoadJobConfig

    logger.info("[SAVE] Writing features to %s…", table_ref)
    job_config = LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    logger.info("[OK] Feature table written successfully")

    return table_ref
