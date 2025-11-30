"""Federated server implementing FedAvg for the FedIoT workflow."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import FederatedConfig
from src.data.dataset import DatasetBundle

from .client import Client, ClientUpdate
from .metrics import compute_metrics


@dataclass
class RoundReport:
    """Aggregate information about a single federated round."""

    round_id: int
    client_losses: Dict[int, float]
    global_metrics: Optional[Dict[str, float]] = None


class FederatedServer:
    """Coordinate federated rounds and global evaluation."""

    def __init__(
        self,
        model: nn.Module,
        clients: Iterable[Client],
        config: FederatedConfig,
        dataset_bundle: DatasetBundle,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.clients = list(clients)
        self.config = config
        self.dataset_bundle = dataset_bundle
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        random.seed(self.config.seed)

    def _sample_clients(self) -> List[Client]:
        return random.sample(self.clients, k=self.config.clients_per_round)

    def _aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        total_samples = sum(update.num_samples for update in updates)
        aggregated_state: Dict[str, torch.Tensor] = {}
        for update in updates:
            weight = update.num_samples / total_samples
            for name, tensor in update.weights.items():
                if name not in aggregated_state:
                    aggregated_state[name] = tensor.clone() * weight
                else:
                    aggregated_state[name] += tensor * weight
        return aggregated_state

    def _update_global_model(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def _reconstruction_errors(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        data_loader = DataLoader(
            TensorDataset(torch.from_numpy(features).float()),
            batch_size=batch_size,
            shuffle=False,
        )
        self.model.eval()
        errors: List[float] = []
        with torch.no_grad():
            for (batch,) in data_loader:
                batch = batch.to(self.device)
                reconstruction = self.model(batch)
                batch_errors = torch.mean((reconstruction - batch) ** 2, dim=1)
                errors.extend(batch_errors.cpu().tolist())
        return np.array(errors)

    def _evaluate(self) -> Dict[str, float]:
        train_errors = self._reconstruction_errors(self.dataset_bundle.x_train)
        val_errors = self._reconstruction_errors(self.dataset_bundle.x_val)
        threshold, val_metrics = self._select_threshold(
            val_errors,
            self.dataset_bundle.y_val,
            train_errors,
        )
        test_errors = self._reconstruction_errors(self.dataset_bundle.x_test)
        metrics = compute_metrics(
            y_true=self.dataset_bundle.y_test,
            reconstruction_errors=test_errors,
            threshold=threshold,
        )
        metrics["train_error_mean"] = float(train_errors.mean())
        metrics["test_error_mean"] = float(test_errors.mean())
        metrics["threshold"] = threshold
        for key, value in val_metrics.items():
            metrics[f"val_{key}"] = value
        return metrics

    def _select_threshold(
        self,
        val_errors: np.ndarray,
        val_labels: np.ndarray,
        train_errors: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        metric_key = self.config.threshold_metric.lower()
        quantiles = np.linspace(0.0, 1.0, num=max(self.config.threshold_grid_size, 2))
        candidate_thresholds = np.unique(np.quantile(val_errors, quantiles))
        candidate_thresholds = np.array(
            [float(t) for t in candidate_thresholds if np.isfinite(t) and t > 0.0]
        )

        best_threshold: Optional[float] = None
        best_score = float("-inf")
        best_metrics: Optional[Dict[str, float]] = None

        for threshold in candidate_thresholds:
            metrics = compute_metrics(
                y_true=val_labels,
                reconstruction_errors=val_errors,
                threshold=float(threshold),
            )
            if metrics["recall"] < self.config.threshold_min_recall:
                continue
            if metrics["precision"] < self.config.threshold_min_precision:
                continue
            if metrics.get("fpr", 0.0) > self.config.threshold_max_fpr:
                continue
            score = metrics.get(metric_key, float("nan"))
            if not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics

        if best_threshold is None or best_metrics is None:
            fallback_threshold = float(
                np.quantile(train_errors, self.config.anomaly_threshold_quantile)
            )
            fallback_metrics = compute_metrics(
                y_true=val_labels,
                reconstruction_errors=val_errors,
                threshold=fallback_threshold,
            )
            fallback_metrics["threshold"] = fallback_threshold
            return fallback_threshold, fallback_metrics

        best_metrics["threshold"] = best_threshold
        return best_threshold, best_metrics

    def train(self) -> List[RoundReport]:
        """Run the configured number of federated rounds."""

        history: List[RoundReport] = []
        for round_id in range(1, self.config.rounds + 1):
            selected_clients = self._sample_clients()
            updates: List[ClientUpdate] = []
            client_losses: Dict[int, float] = {}

            for client in selected_clients:
                update = client.train(self.model)
                updates.append(update)
                client_losses[client.client_id] = update.stats.final_loss

            aggregated_state = self._aggregate(updates)
            self._update_global_model(aggregated_state)

            global_metrics: Optional[Dict[str, float]] = None
            if round_id % self.config.evaluation_round_interval == 0:
                global_metrics = self._evaluate()

            history.append(
                RoundReport(
                    round_id=round_id,
                    client_losses=client_losses,
                    global_metrics=global_metrics,
                )
            )
        return history
