"""Visualize ROC and precision-recall curves for the trained FedIoT autoencoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_iot_dataset
from src.federated.metrics import compute_metrics
from src.federated.server import FederatedServer
from src.models import Autoencoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC and PR curves for the trained FedIoT model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration YAML",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/global_model.pt"),
        help="Path to the trained global model weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/roc_pr_curves.png"),
        help="Destination for the saved figure",
    )
    return parser.parse_args()


def _prepare_artifacts(
    config_path: Path, model_path: Path
) -> Tuple[FederatedServer, float, np.ndarray, np.ndarray, np.ndarray]:
    experiment_config = load_config(config_path)
    dataset_bundle = load_iot_dataset(experiment_config.data)

    experiment_config.model.input_dim = dataset_bundle.x_train.shape[1]
    model = Autoencoder(experiment_config.model)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    server = FederatedServer(
        model=model,
        clients=[],
        config=experiment_config.federated,
        dataset_bundle=dataset_bundle,
    )

    train_errors = server._reconstruction_errors(dataset_bundle.x_train)
    val_errors = server._reconstruction_errors(dataset_bundle.x_val)
    test_errors = server._reconstruction_errors(dataset_bundle.x_test)
    threshold, _ = server._select_threshold(
        val_errors,
        dataset_bundle.y_val,
        train_errors,
    )
    return server, threshold, train_errors, val_errors, test_errors


def _threshold_confusion(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predictions = (errors >= threshold).astype(int)
    tp = float(np.logical_and(predictions == 1, labels == 1).sum())
    fp = float(np.logical_and(predictions == 1, labels == 0).sum())
    tn = float(np.logical_and(predictions == 0, labels == 0).sum())
    fn = float(np.logical_and(predictions == 0, labels == 1).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tnr": tnr,
        "accuracy": accuracy,
        "f1": f1,
    }


def _plot_curves(
    errors: np.ndarray,
    labels: np.ndarray,
    axes: Tuple[plt.Axes, plt.Axes],
    label_prefix: str,
    threshold: float,
) -> Tuple[float, float, Dict[str, float]]:
    roc_axis, pr_axis = axes

    fpr, tpr, _ = roc_curve(labels, errors)
    precision, recall, _ = precision_recall_curve(labels, errors)

    roc_auc = roc_auc_score(labels, errors)
    pr_auc = average_precision_score(labels, errors)

    roc_axis.plot(fpr, tpr, label=f"{label_prefix} ROC (AUC={roc_auc:.3f})")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.set_xlim(0.0, 1.0)
    roc_axis.set_ylim(0.0, 1.05)
    roc_axis.grid(True, linestyle=":", linewidth=0.7)

    pr_axis.plot(recall, precision, label=f"{label_prefix} PR (AP={pr_auc:.3f})")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_xlim(0.0, 1.0)
    pr_axis.set_ylim(0.0, 1.05)
    pr_axis.grid(True, linestyle=":", linewidth=0.7)

    threshold_metrics = _threshold_confusion(errors, labels, threshold)
    roc_axis.scatter(
        threshold_metrics["fpr"],
        threshold_metrics["recall"],
        color="red",
        marker="o",
        label="Selected threshold",
    )
    pr_axis.scatter(
        threshold_metrics["recall"],
        threshold_metrics["precision"],
        color="red",
        marker="o",
        label="Selected threshold",
    )

    roc_axis.legend(loc="lower right")
    pr_axis.legend(loc="lower left")

    return roc_auc, pr_auc, threshold_metrics


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    server, threshold, _, val_errors, test_errors = _prepare_artifacts(args.config, args.model)
    bundle = server.dataset_bundle

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    val_roc_auc, val_pr_auc, val_threshold_metrics = _plot_curves(
        errors=val_errors,
        labels=bundle.y_val,
        axes=(axes[0, 0], axes[1, 0]),
        label_prefix="Validation",
        threshold=threshold,
    )

    test_roc_auc, test_pr_auc, test_threshold_metrics = _plot_curves(
        errors=test_errors,
        labels=bundle.y_test,
        axes=(axes[0, 1], axes[1, 1]),
        label_prefix="Test",
        threshold=threshold,
    )

    axes[0, 0].set_title(f"Validation ROC (AUC={val_roc_auc:.3f})")
    axes[0, 1].set_title(f"Test ROC (AUC={test_roc_auc:.3f})")
    axes[1, 0].set_title(f"Validation PR (AP={val_pr_auc:.3f})")
    axes[1, 1].set_title(f"Test PR (AP={test_pr_auc:.3f})")

    fig.suptitle(
        "FedIoT Autoencoder ROC & PR Curves\n"
        f"Threshold={threshold:.4f} | Val AUC={val_roc_auc:.3f}/{val_pr_auc:.3f} | "
        f"Test AUC={test_roc_auc:.3f}/{test_pr_auc:.3f}"
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    fig.savefig(args.output, dpi=200)
    print(f"Saved ROC/PR visualization to {args.output}")

    val_metrics = compute_metrics(bundle.y_val, val_errors, threshold)
    test_metrics = compute_metrics(bundle.y_test, test_errors, threshold)

    print("Validation metrics at selected threshold:", val_metrics)
    print("Validation confusion-derived metrics:", val_threshold_metrics)
    print("Test metrics at selected threshold:", test_metrics)
    print("Test confusion-derived metrics:", test_threshold_metrics)


if __name__ == "__main__":
    main()
