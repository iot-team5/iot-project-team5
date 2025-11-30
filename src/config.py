"""Configuration helpers for the FedIoT reference implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Dataset and partitioning configuration."""

    dataset_path: Path
    feature_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    log_transform_columns: Optional[List[str]] = None
    target_column: str = "label"
    positive_label: int = 1
    negative_label: int = 0
    train_label_filter: Optional[List[int]] = None
    test_split: float = 0.2
    validation_split: float = 0.1
    num_clients: int = 5
    min_samples_per_client: int = 128
    iid: bool = False
    seed: int = 42


@dataclass
class TrainingConfig:
    """Hyperparameters for local training."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 3
    early_stopping_delta: float = 1e-4
    latent_noise_std: float = 0.0


@dataclass
class ModelConfig:
    """Autoencoder architecture definition."""

    input_dim: int
    encoder_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    decoder_layers: List[int] = field(default_factory=lambda: [64, 128])
    latent_dim: int = 16
    dropout: float = 0.1
    activation: str = "relu"
    latent_normalization: bool = False


@dataclass
class FederatedConfig:
    """Federated learning orchestration parameters."""

    rounds: int = 10
    clients_per_round: int = 3
    aggregation: str = "fedavg"
    evaluation_round_interval: int = 1
    seed: int = 42
    anomaly_threshold_quantile: float = 0.95
    threshold_metric: str = "f1"
    threshold_min_recall: float = 0.0
    threshold_min_precision: float = 0.0
    threshold_grid_size: int = 200
    threshold_max_fpr: float = 1.0


@dataclass
class ExperimentConfig:
    """Top level container for experiment configuration."""

    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    federated: FederatedConfig
    output_dir: Path = Path("outputs")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ExperimentConfig":
        """Instantiate the config from a plain dictionary."""

        data_cfg = DataConfig(**config_dict["data"])
        model_cfg = ModelConfig(**config_dict["model"])
        training_cfg = TrainingConfig(**config_dict["training"])
        federated_cfg = FederatedConfig(**config_dict["federated"])
        output_dir = Path(config_dict.get("output_dir", "outputs"))
        return cls(
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
            federated=federated_cfg,
            output_dir=output_dir,
        )


def load_config(path: Path) -> ExperimentConfig:
    """Load the experiment configuration from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return ExperimentConfig.from_dict(raw_config)
