"""KFP component -- data profiling (train vs test split comparison)."""

from kfp import dsl

from fraud_detector.pipelines import pipeline_component


@pipeline_component()
def data_profile_op(
    project_id: str,
    bq_dataset: str,
    feature_table: str,
    split_date: str,
    read_features_sql: str,
    profile_report: dsl.Output[dsl.HTML],
) -> None:
    """Generate ydata-profiling comparison report for train/test splits."""
    import logging
    import os

    import pandas as pd
    from google.cloud import bigquery
    from ydata_profiling import ProfileReport

    from fraud_detector import FraudDetector

    logger = logging.getLogger(__name__)

    logger.info("-" * 60)
    logger.info("[DATA] STEP: Data Profiling")
    logger.info("-" * 60)

    client = bigquery.Client(project=project_id)
    query = read_features_sql.format(
        project_id=project_id,
        bq_dataset=bq_dataset,
        feature_table=feature_table,
    )
    df = client.query(query).to_dataframe()
    df["tx_ts"] = pd.to_datetime(df["tx_ts"], utc=True).dt.tz_localize(None)

    train_df, test_df = FraudDetector.split(df, split_date)
    logger.info("[DATA] Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Drop non-feature columns for profiling
    drop_cols = [c for c in ("tx_id", "tx_ts", "customer_id", "terminal_id") if c in train_df.columns]
    train_profile = train_df.drop(columns=drop_cols)
    test_profile = test_df.drop(columns=drop_cols)

    train_report = ProfileReport(train_profile, title="Train Split", minimal=True)
    test_report = ProfileReport(test_profile, title="Test Split", minimal=True)
    comparison = train_report.compare(test_report)

    os.makedirs(os.path.dirname(profile_report.path), exist_ok=True)
    comparison.to_file(profile_report.path)
    logger.info("[OK] Profile comparison report written to %s", profile_report.path)
