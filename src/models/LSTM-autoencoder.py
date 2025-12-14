"""LSTM Autoencoder model (FedIoT-style)."""

from __future__ import annotations

from typing import Callable
import torch
from torch import nn

from src.config import ModelConfig


def _resolve_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder
    Input  : (B, T, F)
    Output : (B, T, F)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.input_dim = config.input_dim
        self.seq_len = config.seq_len
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        activation_factory = lambda: _resolve_activation(config.activation)

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.latent_layer = nn.Linear(config.hidden_dim, config.latent_dim)
        self.latent_norm = (
            nn.LayerNorm(config.latent_dim)
            if config.latent_normalization
            else None
        )
        self.latent_activation = activation_factory()
        self._latent_noise_std = 0.0

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder_input = nn.Linear(config.latent_dim, config.hidden_dim)

        self.decoder = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(config.hidden_dim, config.input_dim)

    # -------------------------
    # Latent noise (optional)
    # -------------------------
    def set_latent_noise(self, std: float) -> None:
        self._latent_noise_std = max(0.0, float(std))

    def _apply_latent_noise(self, latent: torch.Tensor) -> torch.Tensor:
        if self.training and self._latent_noise_std > 0.0:
            noise = torch.randn_like(latent) * self._latent_noise_std
            latent = latent + noise
        return latent

    # -------------------------
    # Encode
    # -------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return latent: (B, latent_dim)
        """
        _, (h_n, _) = self.encoder(x)
        # 마지막 레이어의 마지막 timestep hidden
        h_last = h_n[-1]

        latent = self.latent_layer(h_last)
        if self.latent_norm is not None:
            latent = self.latent_norm(latent)

        return self.latent_activation(latent)

    # -------------------------
    # Decode
    # -------------------------
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, latent_dim)
        return reconstruction: (B, T, F)
        """
        latent = self._apply_latent_noise(latent)

        # (B, hidden_dim)
        decoder_input = self.decoder_input(latent)

        # 시계열 길이만큼 복제
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)

        dec_out, _ = self.decoder(decoder_input)
        return self.output_layer(dec_out)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)