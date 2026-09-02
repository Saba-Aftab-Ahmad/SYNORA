"""
SBFLT-24 Subtask 1: FL Experiment Configuration
Defines schema and validation for all configurable
federated learning experiment parameters.

Author: Kashaf Kamran
Sprint: 5
"""

import json
import os
import time
from typing import Any, Dict, List, Optional


# Default configuration values
DEFAULT_CONFIG = {
    "experiment_name": "synora_fl_experiment",
    "num_rounds": 10,
    "num_clients": 3,
    "min_clients_per_round": 2,
    "learning_rate": 0.01,
    "batch_size": 32,
    "local_epochs": 5,
    "partition_type": "non_iid",
    "dirichlet_alpha": 0.5,
    "vocab_size": 5000,
    "max_sequence_length": 100,
    "embedding_dim": 64,
    "num_classes": 3,
    "languages": ["dholuo", "kalenjin", "kidawida"],
    "save_checkpoints": True,
    "checkpoint_dir": "data/checkpoints",
    "log_dir": "data/logs",
    "seed": 42
}

# Valid options for categorical parameters
VALID_PARTITION_TYPES = ["iid", "non_iid"]
VALID_LANGUAGES = ["dholuo", "kalenjin", "kidawida"]


class ConfigValidationError(Exception):
    """
    Raised when configuration validation fails.
    Contains field name and reason for failure.
    """
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(
            f"Config error in '{field}': {message}"
        )


class ExperimentConfig:
    """
    Manages FL experiment configuration parameters.
    Validates all inputs and provides typed access.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise with provided or default config.

        Args:
            config (dict): Configuration parameters.
                          Missing keys use defaults.
        """
        # Start with defaults
        self.config = DEFAULT_CONFIG.copy()

        # Override with provided values
        if config:
            self.config.update(config)

        # Validate on initialisation
        self.validate()

        print(
            f"ExperimentConfig initialised: "
            f"'{self.config['experiment_name']}'"
        )

    def validate(self):
        """
        Validate all configuration parameters.
        Raises ConfigValidationError on any failure.
        """
        self._validate_experiment_name()
        self._validate_num_rounds()
        self._validate_num_clients()
        self._validate_min_clients()
        self._validate_learning_rate()
        self._validate_batch_size()
        self._validate_local_epochs()
        self._validate_partition_type()
        self._validate_dirichlet_alpha()
        self._validate_vocab_size()
        self._validate_sequence_length()
        self._validate_embedding_dim()
        self._validate_num_classes()
        self._validate_languages()
        self._validate_seed()

        print("✅ Configuration validation passed")

    def _validate_experiment_name(self):
        name = self.config.get("experiment_name", "")
        if not isinstance(name, str) or not name.strip():
            raise ConfigValidationError(
                "experiment_name",
                "Must be a non-empty string"
            )

    def _validate_num_rounds(self):
        value = self.config.get("num_rounds")
        if not isinstance(value, int) or value <= 0:
            raise ConfigValidationError(
                "num_rounds",
                f"Must be positive integer, got {value}"
            )
        if value > 1000:
            raise ConfigValidationError(
                "num_rounds",
                f"Cannot exceed 1000, got {value}"
            )

    def _validate_num_clients(self):
        value = self.config.get("num_clients")
        if not isinstance(value, int) or value < 2:
            raise ConfigValidationError(
                "num_clients",
                f"Must be at least 2, got {value}"
            )

    def _validate_min_clients(self):
        min_c = self.config.get("min_clients_per_round")
        num_c = self.config.get("num_clients")
        if not isinstance(min_c, int) or min_c < 1:
            raise ConfigValidationError(
                "min_clients_per_round",
                f"Must be positive integer, got {min_c}"
            )
        if min_c > num_c:
            raise ConfigValidationError(
                "min_clients_per_round",
                f"Cannot exceed num_clients "
                f"({num_c}), got {min_c}"
            )

    def _validate_learning_rate(self):
        value = self.config.get("learning_rate")
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(
                "learning_rate",
                "Must be a number"
            )
        if value <= 0:
            raise ConfigValidationError(
                "learning_rate",
                f"Must be positive, got {value}"
            )
        if value > 1.0:
            raise ConfigValidationError(
                "learning_rate",
                f"Cannot exceed 1.0, got {value}"
            )

    def _validate_batch_size(self):
        value = self.config.get("batch_size")
        if not isinstance(value, int) or value < 1:
            raise ConfigValidationError(
                "batch_size",
                f"Must be positive integer, got {value}"
            )

    def _validate_local_epochs(self):
        value = self.config.get("local_epochs")
        if not isinstance(value, int) or value < 1:
            raise ConfigValidationError(
                "local_epochs",
                f"Must be positive integer, got {value}"
            )

    def _validate_partition_type(self):
        value = self.config.get("partition_type")
        if value not in VALID_PARTITION_TYPES:
            raise ConfigValidationError(
                "partition_type",
                f"Must be one of {VALID_PARTITION_TYPES}, "
                f"got '{value}'"
            )

    def _validate_dirichlet_alpha(self):
        value = self.config.get("dirichlet_alpha")
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(
                "dirichlet_alpha",
                "Must be a number"
            )
        if value <= 0:
            raise ConfigValidationError(
                "dirichlet_alpha",
                f"Must be positive, got {value}"
            )

    def _validate_vocab_size(self):
        value = self.config.get("vocab_size")
        if not isinstance(value, int) or value < 100:
            raise ConfigValidationError(
                "vocab_size",
                f"Must be at least 100, got {value}"
            )

    def _validate_sequence_length(self):
        value = self.config.get("max_sequence_length")
        if not isinstance(value, int) or value < 10:
            raise ConfigValidationError(
                "max_sequence_length",
                f"Must be at least 10, got {value}"
            )

    def _validate_embedding_dim(self):
        value = self.config.get("embedding_dim")
        if not isinstance(value, int) or value < 8:
            raise ConfigValidationError(
                "embedding_dim",
                f"Must be at least 8, got {value}"
            )

    def _validate_num_classes(self):
        value = self.config.get("num_classes")
        if not isinstance(value, int) or value < 2:
            raise ConfigValidationError(
                "num_classes",
                f"Must be at least 2, got {value}"
            )

    def _validate_languages(self):
        value = self.config.get("languages", [])
        if not isinstance(value, list) or len(value) == 0:
            raise ConfigValidationError(
                "languages",
                "Must be a non-empty list"
            )
        for lang in value:
            if lang not in VALID_LANGUAGES:
                raise ConfigValidationError(
                    "languages",
                    f"'{lang}' not in {VALID_LANGUAGES}"
                )

    def _validate_seed(self):
        value = self.config.get("seed")
        if not isinstance(value, int):
            raise ConfigValidationError(
                "seed",
                f"Must be an integer, got {value}"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)

    def to_dict(self) -> Dict:
        """Return full configuration as dictionary."""
        return self.config.copy()

    def summary(self):
        """Print a formatted configuration summary."""
        print("\n" + "=" * 50)
        print("FL EXPERIMENT CONFIGURATION")
        print("=" * 50)
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")