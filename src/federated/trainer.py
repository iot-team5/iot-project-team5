"""Local training utilities for FedIoT clients."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TrainingConfig


@dataclass
class TrainingStats:
    """Summary of a local training run."""

    epochs_ran: int
    final_loss: float


class LocalTrainer:
    """Perform local autoencoder updates on a single client."""

    def __init__(self, config: TrainingConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_loader(self, x_train: np.ndarray) -> DataLoader:
        features = torch.from_numpy(x_train).float()
        dataset = TensorDataset(features)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def train(
        self,
        model: nn.Module,
        x_train: np.ndarray,
    ) -> Tuple[Dict[str, torch.Tensor], TrainingStats]:
        """Execute local training and return updated weights and stats."""

        local_model = copy.deepcopy(model).to(self.device)
        local_model.train()

        if hasattr(local_model, "set_latent_noise"):
            local_model.set_latent_noise(self.config.latent_noise_std)

        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        epochs_without_improvement = 0
        last_loss = best_loss

        data_loader = self._prepare_loader(x_train)

        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            for (batch_features,) in data_loader:
                batch_features = batch_features.to(self.device)
                optimizer.zero_grad()
                reconstructions = local_model(batch_features)
                loss = criterion(reconstructions, batch_features)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_features.size(0)

            epoch_loss /= len(data_loader.dataset)

            if epoch_loss + self.config.early_stopping_delta < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            last_loss = epoch_loss
            if epochs_without_improvement >= self.config.early_stopping_patience:
                break

        state_dict = {k: v.cpu() for k, v in local_model.state_dict().items()}
        if hasattr(local_model, "set_latent_noise"):
            local_model.set_latent_noise(0.0)
        stats = TrainingStats(epochs_ran=epoch, final_loss=last_loss)
        return state_dict, stats
