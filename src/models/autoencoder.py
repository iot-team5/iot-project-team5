"""Deep autoencoder model used in the FedIoT baseline."""

from __future__ import annotations

from typing import Callable, Iterable, List

import torch
from torch import nn

from src.config import ModelConfig


def _make_mlp(
    input_dim: int,
    layer_sizes: Iterable[int],
    activation_factory: Callable[[], nn.Module],
    dropout: float,
) -> nn.Sequential:
    """Utility to build a stacked MLP with dropout."""

    layers: List[nn.Module] = []
    prev_dim = input_dim
    for size in layer_sizes:
        layers.append(nn.Linear(prev_dim, size))
        layers.append(activation_factory())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = size
    return nn.Sequential(*layers)


def _resolve_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "gelu": nn.GELU(),
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


class Autoencoder(nn.Module):
    """Deep autoencoder with configurable encoder and decoder widths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        activation_factory = lambda: _resolve_activation(config.activation)
        self.encoder = _make_mlp(
            input_dim=config.input_dim,
            layer_sizes=config.encoder_layers,
            activation_factory=activation_factory,
            dropout=config.dropout,
        )
        encoder_output = config.encoder_layers[-1] if config.encoder_layers else config.input_dim
        self.latent_layer = nn.Linear(encoder_output, config.latent_dim)
        self.latent_norm = nn.LayerNorm(config.latent_dim) if config.latent_normalization else None
        self.latent_activation = activation_factory()
        self._latent_noise_std = 0.0

        decoder_input = config.latent_dim
        decoder_layers = list(config.decoder_layers)
        if not decoder_layers or decoder_layers[-1] != config.input_dim:
            decoder_layers.append(config.input_dim)
        self.decoder = _make_mlp(
            input_dim=decoder_input,
            layer_sizes=decoder_layers,
            activation_factory=activation_factory,
            dropout=config.dropout,
        )

    def set_latent_noise(self, std: float) -> None:
        self._latent_noise_std = max(0.0, float(std))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(inputs)
        latent = self.latent_layer(hidden)
        if self.latent_norm is not None:
            latent = self.latent_norm(latent)
        return self.latent_activation(latent)

    def _apply_latent_noise(self, latent: torch.Tensor) -> torch.Tensor:
        if self.training and self._latent_noise_std > 0.0:
            noise = torch.randn_like(latent) * self._latent_noise_std
            latent = latent + noise
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        latent = self._apply_latent_noise(latent)
        return self.decode(latent)
