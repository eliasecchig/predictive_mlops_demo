"""Load FraudFinder data into BigQuery.

Supports two modes:
  1. Load from GCS parquet files (--source gcs)
  2. Generate synthetic data for testing (--source synthetic)
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DATA_BUCKET = "gs://fraudfinder-public-data"

# Sample: ~100K rows (1 day), ~11 MB — fast iteration
DATA_SOURCE_TRANSACTIONS_SAMPLE = f"{DATA_BUCKET}/sample/tx/*.parquet"
DATA_SOURCE_LABELS_SAMPLE = f"{DATA_BUCKET}/sample/tx_labels/*.parquet"

# Full: ~3.1M rows (1 month), ~308 MB
DATA_SOURCE_TRANSACTIONS_FULL = f"{DATA_BUCKET}/tx/*.parquet"
DATA_SOURCE_LABELS_FULL = f"{DATA_BUCKET}/tx_labels/*.parquet"


def create_dataset(client: bigquery.Client, project_id: str, dataset_id: str, location: str = "US") -> None:
    """Create a BigQuery dataset if it doesn't exist."""
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        logger.info("Dataset %s already exists", dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        logger.info("Created dataset %s", dataset_ref)


def load_parquet_to_bq(
    client: bigquery.Client,
    gcs_uri: str,
    table_ref: str,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    """Load a parquet file from GCS into a BigQuery table."""
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=write_disposition,
    )
    logger.info("Loading %s → %s", gcs_uri, table_ref)
    job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    job.result()
    table = client.get_table(table_ref)
    logger.info("Loaded %d rows into %s", table.num_rows, table_ref)


def generate_synthetic_data(n_transactions: int = 10_000, fraud_rate: float = 0.012, seed: int = 42):
    """Generate synthetic FraudFinder-like data."""
    rng = np.random.RandomState(seed)

    n_customers = 5000
    n_terminals = 200

    # Generate transactions spanning Jan 2024 (matches train_test_split_date)
    tx_ids = list(range(n_transactions))
    start_date = pd.Timestamp("2024-01-01")
    tx_ts = [start_date + pd.Timedelta(hours=rng.randint(0, 31 * 24)) for _ in range(n_transactions)]
    tx_ts.sort()

    customer_ids = rng.randint(0, n_customers, size=n_transactions)
    terminal_ids = rng.randint(0, n_terminals, size=n_transactions)

    # Amount: mostly small, some large
    tx_amounts = np.round(rng.exponential(scale=50, size=n_transactions), 2)
    tx_amounts = np.clip(tx_amounts, 0.01, 5000.0)

    tx_df = pd.DataFrame({
        "tx_id": tx_ids,
        "tx_ts": tx_ts,
        "customer_id": customer_ids,
        "terminal_id": terminal_ids,
        "tx_amount": tx_amounts,
    })

    # Generate fraud labels — higher amounts and certain terminals are more likely to be fraud
    fraud_prob = np.full(n_transactions, fraud_rate)
    # High amounts more likely fraud
    fraud_prob[tx_amounts > 200] *= 5
    fraud_prob[tx_amounts > 500] *= 3
    # Certain terminals are compromised
    compromised = rng.choice(n_terminals, size=10, replace=False)
    for t in compromised:
        fraud_prob[terminal_ids == t] *= 8
    fraud_prob = np.clip(fraud_prob, 0, 0.5)

    tx_fraud = rng.binomial(1, fraud_prob)

    labels_df = pd.DataFrame({
        "tx_id": tx_ids,
        "tx_fraud": tx_fraud,
    })

    logger.info("Generated %d transactions (%.1f%% fraud)", n_transactions, tx_fraud.mean() * 100)
    return tx_df, labels_df


def load_df_to_bq(client: bigquery.Client, df: pd.DataFrame, table_ref: str) -> None:
    """Load a DataFrame into BigQuery."""
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    logger.info("Loading %d rows → %s", len(df), table_ref)
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    table = client.get_table(table_ref)
    logger.info("Loaded %d rows into %s", table.num_rows, table_ref)


def main():
    parser = argparse.ArgumentParser(description="Load FraudFinder data into BigQuery")
    parser.add_argument("--project-id", default=None, help="GCP project ID (defaults to $PROJECT_ID or gcloud config)")
    parser.add_argument("--dataset", default="fraud_detection", help="BigQuery dataset name")
    parser.add_argument("--location", default="US", help="BigQuery dataset location")
    parser.add_argument(
        "--source", choices=["gcs", "synthetic"], default="gcs",
        help="Data source: 'gcs' for parquet files, 'synthetic' for generated data",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full 1-month dataset (~3.1M rows) instead of sample (~100K rows)",
    )
    parser.add_argument("--n-transactions", type=int, default=50_000, help="Number of synthetic transactions")
    args = parser.parse_args()

    if args.project_id is None:
        from fraud_detector.config import get_project_id

        args.project_id = get_project_id()
    client = bigquery.Client(project=args.project_id)

    # Step 1: Create dataset
    logger.info("=== Step 1: Create BigQuery dataset ===")
    create_dataset(client, args.project_id, args.dataset, args.location)

    tx_table = f"{args.project_id}.{args.dataset}.tx"
    labels_table = f"{args.project_id}.{args.dataset}.txlabels"

    if args.source == "gcs":
        if args.full:
            tx_uri = DATA_SOURCE_TRANSACTIONS_FULL
            labels_uri = DATA_SOURCE_LABELS_FULL
            logger.info("Using FULL dataset (~3.1M rows)")
        else:
            tx_uri = DATA_SOURCE_TRANSACTIONS_SAMPLE
            labels_uri = DATA_SOURCE_LABELS_SAMPLE
            logger.info("Using SAMPLE dataset (~100K rows). Use --full for the complete dataset.")

        # Step 2: Load from GCS
        logger.info("=== Step 2: Load transactions from GCS ===")
        load_parquet_to_bq(client, tx_uri, tx_table)

        logger.info("=== Step 3: Load labels from GCS ===")
        load_parquet_to_bq(client, labels_uri, labels_table)

    else:
        # Step 2: Generate and load synthetic data
        logger.info("=== Step 2: Generate synthetic data ===")
        tx_df, labels_df = generate_synthetic_data(n_transactions=args.n_transactions)

        logger.info("=== Step 3: Load into BigQuery ===")
        load_df_to_bq(client, tx_df, tx_table)
        load_df_to_bq(client, labels_df, labels_table)

    # Step 4: Verify
    logger.info("=== Step 4: Verify ===")
    for table_name in ["tx", "txlabels"]:
        ref = f"{args.project_id}.{args.dataset}.{table_name}"
        table = client.get_table(ref)
        logger.info("  %s: %d rows", ref, table.num_rows)

    query = f"""
        SELECT t.tx_id, t.tx_ts, t.customer_id, t.terminal_id, t.tx_amount, l.tx_fraud
        FROM `{args.project_id}.{args.dataset}.tx` t
        LEFT JOIN `{args.project_id}.{args.dataset}.txlabels` l ON t.tx_id = l.tx_id
        ORDER BY t.tx_ts
        LIMIT 5
    """
    logger.info("  Sample joined data:")
    for row in client.query(query).result():
        logger.info("    %s", dict(row))

    # Fraud stats
    stats_query = f"""
        SELECT
            COUNT(*) as total,
            SUM(l.tx_fraud) as fraud_count,
            ROUND(AVG(l.tx_fraud) * 100, 2) as fraud_pct
        FROM `{args.project_id}.{args.dataset}.tx` t
        JOIN `{args.project_id}.{args.dataset}.txlabels` l ON t.tx_id = l.tx_id
    """
    for row in client.query(stats_query).result():
        logger.info("  Total: %d, Fraud: %d (%.2f%%)", row.total, row.fraud_count, row.fraud_pct)

    logger.info("=== Data setup complete ===")


if __name__ == "__main__":
    main()
