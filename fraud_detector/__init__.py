"""Fraud Detector -- production Python package for the E2E Predictive MLOps Demo."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s -- %(name)s -- %(message)s",
)

from fraud_detector.config import load_config, load_sql  # noqa: E402
from fraud_detector.model import FraudDetector  # noqa: E402

__all__ = ["FraudDetector", "load_config", "load_sql"]
