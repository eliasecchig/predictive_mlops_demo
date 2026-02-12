"""FraudDetector -- pure ML class: feature engineering, training, evaluation, prediction."""

import logging
from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class FraudDetector:
    """Fraud detection ML logic -- no I/O, no GCP dependencies.

    I/O (BigQuery, GCS, Vertex AI) is handled by pipeline components.

    Usage::

        fd = FraudDetector()
        features = FraudDetector.compute_features(raw_df)
        fd.train(train_df)
        fd.evaluate(test_df)
        print(fd.metrics["auc_roc"])
        scored = fd.predict(new_df)
    """

    ROLLING_WINDOWS: ClassVar[list[int]] = [1, 7, 28, 90]

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.metrics: dict | None = None

    # -- Features --------------------------------------------------------

    @staticmethod
    def feature_columns(windows: list[int] | None = None) -> list[str]:
        """Return the list of engineered feature column names."""
        if windows is None:
            windows = FraudDetector.ROLLING_WINDOWS
        cols = ["tx_amount"]
        for group in ["customer", "terminal"]:
            for window in windows:
                for agg in ["count", "avg", "max"]:
                    cols.append(f"{agg}_tx_amount_{window}d_{group}")
        return cols

    @staticmethod
    def compute_features(
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute rolling count / avg / max of tx_amount per customer and terminal.

        Args:
            df: Must contain columns tx_ts, tx_amount, customer_id, terminal_id.
            windows: Rolling window sizes in days (default: 1, 7, 28, 90).

        Returns:
            DataFrame with original columns plus 24 rolling-window features.
        """
        if windows is None:
            windows = FraudDetector.ROLLING_WINDOWS

        for group_col in ["customer_id", "terminal_id"]:
            suffix = group_col.replace("_id", "")
            df = df.sort_values([group_col, "tx_ts"]).reset_index(drop=True)
            indexed = df.set_index("tx_ts")

            for w in windows:
                rolling = indexed.groupby(group_col)["tx_amount"].rolling(f"{w}D", min_periods=1)
                counts = rolling.count().droplevel(0).sort_index()
                avgs = rolling.mean().droplevel(0).sort_index()
                maxs = rolling.max().droplevel(0).sort_index()

                df[f"count_tx_amount_{w}d_{suffix}"] = counts.values
                df[f"avg_tx_amount_{w}d_{suffix}"] = avgs.values
                df[f"max_tx_amount_{w}d_{suffix}"] = maxs.values

        logger.info("[OK] Feature engineering complete. Shape: %s", df.shape)
        return df

    # -- Training --------------------------------------------------------

    @staticmethod
    def split(
        df: pd.DataFrame,
        split_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by date into train and test sets."""
        split_ts = pd.Timestamp(split_date)
        ts_col = df["tx_ts"]
        if ts_col.dt.tz is not None:
            ts_col = ts_col.dt.tz_localize(None)
        train = df[ts_col < split_ts].copy()
        test = df[ts_col >= split_ts].copy()
        logger.info(
            "[SPLIT] Train: %d rows, Test: %d rows (split at %s)",
            len(train),
            len(test),
            split_date,
        )
        return train, test

    def train(
        self,
        train_df: pd.DataFrame,
        xgb_params: dict | None = None,
        label_col: str = "tx_fraud",
        feature_cols: list[str] | None = None,
    ) -> "FraudDetector":
        """Train an XGBoost classifier. Sets self.model."""
        if xgb_params is None:
            xgb_params = {}
        if feature_cols is None:
            feature_cols = self.feature_columns()

        X_train = train_df[feature_cols].fillna(0).astype(float)
        y_train = train_df[label_col]

        self.model = XGBClassifier(**xgb_params)
        self.model.fit(X_train, y_train)

        logger.info(
            "[OK] Model trained with %d features, %d samples",
            len(feature_cols),
            len(X_train),
        )
        return self

    def evaluate(
        self,
        test_df: pd.DataFrame,
        label_col: str = "tx_fraud",
        feature_cols: list[str] | None = None,
    ) -> dict:
        """Evaluate model on test set. Sets self.metrics and returns them."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")
        if feature_cols is None:
            feature_cols = self.feature_columns()

        X_test = test_df[feature_cols].fillna(0).astype(float)
        y_test = test_df[label_col]

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        auc_roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.metrics = {
            "auc_roc": float(auc_roc),
            "precision_fraud": float(report.get("1", report.get("1.0", {})).get("precision", 0)),
            "recall_fraud": float(report.get("1", report.get("1.0", {})).get("recall", 0)),
            "f1_fraud": float(report.get("1", report.get("1.0", {})).get("f1-score", 0)),
            "accuracy": float(report.get("accuracy", 0)),
            "confusion_matrix": cm.tolist(),
            "test_samples": len(y_test),
            "fraud_rate": float(y_test.mean()),
        }

        logger.info(
            "[EVAL] Evaluation -- AUC-ROC: %.4f, Precision: %.4f, Recall: %.4f",
            auc_roc,
            self.metrics["precision_fraud"],
            self.metrics["recall_fraud"],
        )
        return self.metrics

    # -- Scoring ---------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run batch predictions and return DataFrame with scores."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load_model() first.")
        if feature_cols is None:
            feature_cols = self.feature_columns()

        X = df[feature_cols].fillna(0).astype(float)
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)

        df = df.copy()
        df["fraud_probability"] = probabilities
        df["fraud_prediction"] = predictions
        df["scored_at"] = pd.Timestamp.now()

        logger.info("[OK] Batch prediction complete: %d rows scored", len(df))
        return df

    # -- Persistence -----------------------------------------------------

    def save_model(self, path: str) -> str:
        """Save model to a local path using joblib."""
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("[SAVE] Model saved to %s", path)
        return path

    def load_model(self, path: str) -> "FraudDetector":
        """Load model from a local path. Sets self.model."""
        self.model = joblib.load(path)
        logger.info("[PKG] Model loaded from %s", path)
        return self
