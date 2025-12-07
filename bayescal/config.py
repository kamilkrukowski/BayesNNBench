"""Configuration management for the BayesCal project."""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    results_dir: Path = project_root / "experiments" / "results"
    configs_dir: Path = project_root / "experiments" / "configs"

    # Training settings
    seed: int = 42
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3

    # Model settings
    hidden_dim: int = 128
    num_layers: int = 2

    # Bayesian settings
    prior_std: float = 1.0
    posterior_std_init: float = 0.1

    # Evaluation settings
    num_samples: int = 100  # For MC sampling in Bayesian models
    num_bins: int = 10  # For calibration metrics

    def model_post_init(self, __context: Any) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

