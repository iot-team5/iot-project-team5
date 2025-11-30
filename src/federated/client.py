"""Federated client abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from .trainer import LocalTrainer, TrainingStats


@dataclass
class ClientUpdate:
    """Container for a single client update."""

    client_id: int
    num_samples: int
    weights: Dict[str, torch.Tensor]
    stats: TrainingStats


class Client:
    """Wrap data and training logic for a single client device."""

    def __init__(
        self,
        client_id: int,
        trainer: LocalTrainer,
        train_data: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.client_id = client_id
        self.trainer = trainer
        self.train_features = train_data[0]
        self.train_labels = train_data[1]

    @property
    def num_samples(self) -> int:
        return int(self.train_features.shape[0])

    def train(self, global_model: nn.Module) -> ClientUpdate:
        """Run local training and package the result for aggregation."""

        weights, stats = self.trainer.train(global_model, self.train_features)
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=self.num_samples,
            weights=weights,
            stats=stats,
        )
