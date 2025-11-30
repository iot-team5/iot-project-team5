"""Evaluation utilities for anomaly detection."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    y_true: np.ndarray,
    reconstruction_errors: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute accuracy and recall metrics using a reconstruction threshold."""

    predictions = (reconstruction_errors >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, predictions))
    recall = float(recall_score(y_true, predictions, zero_division=0))
    precision = float(precision_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))
    true_positive = float(np.logical_and(predictions == 1, y_true == 1).sum())
    false_positive = float(np.logical_and(predictions == 1, y_true == 0).sum())
    true_negative = float(np.logical_and(predictions == 0, y_true == 0).sum())
    false_negative = float(np.logical_and(predictions == 0, y_true == 1).sum())
    fpr = float(false_positive / (false_positive + true_negative)) if (false_positive + true_negative) else 0.0
    specificity = float(true_negative / (true_negative + false_positive)) if (true_negative + false_positive) else 0.0
    return {
        "accuracy": accuracy,
        "recall": recall,
        "threshold": threshold,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "specificity": specificity,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def select_dynamic_threshold(
    reconstruction_errors: np.ndarray,
    quantile: float,
) -> Tuple[float, np.ndarray]:
    """Estimate an anomaly threshold based on the given quantile."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("Quantile must be between 0 and 1")
    threshold = float(np.quantile(reconstruction_errors, quantile))
    return threshold, (reconstruction_errors >= threshold).astype(int)
