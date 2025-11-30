"""Command-line entry point for running the FedIoT simulation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_iot_dataset, partition_dataset
from src.federated import Client, FederatedServer, LocalTrainer
from src.models import Autoencoder
from src.utils.logging import configure_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FedIoT reference workflow")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the output directory",
    )
    return parser.parse_args()


def _instantiate_clients(
    trainer: LocalTrainer,
    partitions: Dict[int, Any],
) -> List[Client]:
    clients: List[Client] = []
    for client_id, data in partitions.items():
        clients.append(Client(client_id=client_id, trainer=trainer, train_data=data))
    return clients


def main() -> None:
    args = _parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)

    experiment_config = load_config(args.config)
    if args.output is not None:
        experiment_config.output_dir = args.output
    experiment_config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_iot_dataset(experiment_config.data)

    model_config = experiment_config.model
    model_config.input_dim = dataset_bundle.x_train.shape[1]
    model = Autoencoder(model_config)

    trainer = LocalTrainer(experiment_config.training)

    partitions = partition_dataset(
        experiment_config.data,
        dataset_bundle.x_train,
        dataset_bundle.y_train,
    )

    for client_id, (_, client_targets) in partitions.items():
        unique, counts = np.unique(client_targets, return_counts=True)
        label_summary = ", ".join(
            f"{int(label)}: {int(count)}" for label, count in zip(unique, counts)
        )
        logger.info(
            "Client %s -> samples=%d labels=[%s]",
            client_id,
            client_targets.shape[0],
            label_summary or "none",
        )

    clients = _instantiate_clients(trainer, partitions)

    server = FederatedServer(
        model=model,
        clients=clients,
        config=experiment_config.federated,
        dataset_bundle=dataset_bundle,
    )
    history = server.train()

    history_payload: List[Dict[str, Any]] = []
    for report in history:
        if report.global_metrics:
            metrics = report.global_metrics
            message = (
                "Round %d | acc=%.4f recall=%.4f precision=%.4f f1=%.4f threshold=%.4f"
                % (
                    report.round_id,
                    metrics.get("accuracy", float("nan")),
                    metrics.get("recall", float("nan")),
                    metrics.get("precision", float("nan")),
                    metrics.get("f1", float("nan")),
                    metrics.get("threshold", float("nan")),
                )
            )
            if "val_recall" in metrics and "val_f1" in metrics:
                message += " | val_recall=%.4f val_f1=%.4f" % (
                    metrics.get("val_recall", float("nan")),
                    metrics.get("val_f1", float("nan")),
                )
            logger.info(message)
        history_payload.append(
            {
                "round_id": report.round_id,
                "client_losses": report.client_losses,
                "global_metrics": report.global_metrics,
            }
        )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    history_file = experiment_config.output_dir / f"history_{timestamp}.json"
    with history_file.open("w", encoding="utf-8") as handle:
        json.dump(history_payload, handle, indent=2)

    torch.save(model.state_dict(), experiment_config.output_dir / "global_model.pt")


if __name__ == "__main__":
    main()
