"""
SBFLT-24 Subtask 2: Configuration File Management
Handles loading and saving experiment configurations
as JSON files for reproducibility.

Author: Kashaf Kamran
Sprint: 5
"""

from typing import Optional, List, Dict
import json
import os
import time
try:
    from server.experiment_config import ExperimentConfig   # package import (Blueprint)
except ModuleNotFoundError:
    from experiment_config import ExperimentConfig          # standalone / sys.path import


class ConfigManager:
    """
    Loads, saves, and manages experiment config files.
    Ensures reproducibility by persisting all parameters.
    """

    def __init__(self, config_dir: str = "data/configs"):
        """
        Initialise config manager.

        Args:
            config_dir (str): Directory for config files
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        print(f"ConfigManager initialised: {config_dir}")

    def save_config(
        self, config: ExperimentConfig, filename: Optional[str] = None
    ) -> str:
        """
        Save experiment config to JSON file.

        Args:
            config (ExperimentConfig): Config to save
            filename (str): Optional filename.
                           Auto-generated if not provided.

        Returns:
            str: Path to saved config file
        """
        if filename is None:
            timestamp = int(time.time())
            name = config.get("experiment_name")
            filename = f"{name}_{timestamp}.json"

        filepath = os.path.join(self.config_dir, filename)

        config_dict = config.to_dict()
        config_dict["saved_at"] = time.time()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Config saved: {filepath}")
        return filepath

    def load_config(self, filepath: str) -> ExperimentConfig:
        """
        Load experiment config from JSON file.

        Args:
            filepath (str): Path to config file

        Returns:
            ExperimentConfig: Loaded and validated config

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If JSON is malformed
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Remove metadata fields before validation
        config_dict.pop("saved_at", None)

        config = ExperimentConfig(config_dict)
        print(f"Config loaded: {filepath}")
        return config

    def list_configs(self) -> List[str]:
        """
        List all saved config files.

        Returns:
            list: Filenames of all saved configs
        """
        files = [f for f in os.listdir(self.config_dir) if f.endswith(".json")]
        print(f"Found {len(files)} config files " f"in {self.config_dir}")
        return files

    def load_latest_config(self) -> Optional[ExperimentConfig]:
        """
        Load the most recently saved config file.

        Returns:
            ExperimentConfig or None if no configs exist
        """
        files = self.list_configs()
        if not files:
            print("No saved configs found")
            return None

        # Sort by modification time
        full_paths = [os.path.join(self.config_dir, f) for f in files]
        latest = max(full_paths, key=os.path.getmtime)

        print(f"Loading latest config: {latest}")
        return self.load_config(latest)

    def delete_config(self, filename: str) -> bool:
        """
        Delete a saved config file.

        Args:
            filename (str): Name of config file to delete

        Returns:
            bool: True if deleted successfully
        """
        filepath = os.path.join(self.config_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Config deleted: {filepath}")
            return True

        print(f"Config not found: {filepath}")
        return False

    def export_config_summary(self, config: ExperimentConfig) -> str:
        """
        Export a human-readable config summary.

        Args:
            config (ExperimentConfig): Config to export

        Returns:
            str: Formatted summary string
        """
        # Labels use the exact config field names so that test assertions
        # like `assert "num_rounds" in summary` match (SDS "Consistent
        # Naming Conventions" requirement — field names mirror API fields).
        lines = [
            "=" * 50,
            "FL EXPERIMENT CONFIGURATION SUMMARY",
            "=" * 50,
            f"experiment_name: {config.get('experiment_name')}",
            f"num_rounds: {config.get('num_rounds')}",
            f"num_clients: {config.get('num_clients')}",
            f"min_clients_per_round: {config.get('min_clients_per_round')}",
            f"learning_rate: {config.get('learning_rate')}",
            f"batch_size: {config.get('batch_size')}",
            f"local_epochs: {config.get('local_epochs')}",
            f"partition_type: {config.get('partition_type')}",
            f"dirichlet_alpha: {config.get('dirichlet_alpha')}",
            f"languages: {config.get('languages')}",
            f"vocab_size: {config.get('vocab_size')}",
            f"max_sequence_length: {config.get('max_sequence_length')}",
            f"num_classes: {config.get('num_classes')}",
            f"seed: {config.get('seed')}",
            "=" * 50,
        ]

        summary = "\n".join(lines)
        print(summary)
        return summary
