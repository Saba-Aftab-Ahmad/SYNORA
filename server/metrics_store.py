"""
SBFLT-23 Subtask 1: Convergence Metrics Store
Collects and stores per-round training metrics
for convergence curve visualisation.

Author: Kashaf Kamran
Sprint: 6
"""

import json
import os
import time
from typing import Dict, List, Optional


class RoundMetrics:
    """
    Stores metrics for a single federated round.
    """

    def __init__(
        self,
        round_number: int,
        accuracy: float,
        loss: float,
        num_clients: int,
        timestamp: Optional[float] = None
    ):
        """
        Initialise round metrics.

        Args:
            round_number (int): Round number
            accuracy (float): Global model accuracy
            loss (float): Global model loss
            num_clients (int): Clients participated
            timestamp (float): Unix timestamp
        """
        self.round_number = round_number
        self.accuracy = accuracy
        self.loss = loss
        self.num_clients = num_clients
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "num_clients": self.num_clients,
            "timestamp": self.timestamp
        }


class MetricsStore:
    """
    Collects and manages per-round training metrics.
    Provides formatted data for convergence charts.
    """

    def __init__(self):
        """Initialise empty metrics store."""
        self.rounds: List[RoundMetrics] = []
        self.created_at = time.time()
        print("MetricsStore initialised")

    def add_round_metrics(
        self,
        round_number: int,
        accuracy: float,
        loss: float,
        num_clients: int
    ):
        """
        Add metrics for a completed round.

        Args:
            round_number (int): Round number
            accuracy (float): Global accuracy
            loss (float): Global loss
            num_clients (int): Participating clients

        Raises:
            ValueError: If values are out of range
        """
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                f"Accuracy must be 0.0-1.0, "
                f"got {accuracy}"
            )
        if loss < 0:
            raise ValueError(
                f"Loss must be non-negative, "
                f"got {loss}"
            )
        if num_clients < 1:
            raise ValueError(
                f"num_clients must be positive, "
                f"got {num_clients}"
            )

        metrics = RoundMetrics(
            round_number=round_number,
            accuracy=accuracy,
            loss=loss,
            num_clients=num_clients
        )
        self.rounds.append(metrics)

        print(
            f"Round {round_number} metrics stored: "
            f"acc={accuracy:.4f}, loss={loss:.4f}"
        )

    def get_accuracy_series(self) -> List[float]:
        """
        Get accuracy values across all rounds.

        Returns:
            list: Accuracy per round in order
        """
        return [r.accuracy for r in self.rounds]

    def get_loss_series(self) -> List[float]:
        """
        Get loss values across all rounds.

        Returns:
            list: Loss per round in order
        """
        return [r.loss for r in self.rounds]

    def get_round_numbers(self) -> List[int]:
        """
        Get round numbers for x-axis labels.

        Returns:
            list: Round numbers in order
        """
        return [r.round_number for r in self.rounds]

    def get_chart_data(self) -> Dict:
        """
        Get formatted data for Chart.js rendering.

        Returns:
            dict: Chart-ready data structure
        """
        return {
            "labels": self.get_round_numbers(),
            "accuracy": self.get_accuracy_series(),
            "loss": self.get_loss_series(),
            "num_rounds": len(self.rounds)
        }

    def get_latest_metrics(self) -> Optional[Dict]:
        """
        Get metrics from the most recent round.

        Returns:
            dict: Latest round metrics or None
        """
        if not self.rounds:
            return None
        return self.rounds[-1].to_dict()

    def get_all_metrics(self) -> List[Dict]:
        """
        Get all round metrics as list of dicts.

        Returns:
            list: All round metrics
        """
        return [r.to_dict() for r in self.rounds]

    def is_accuracy_improving(self) -> bool:
        """
        Check if accuracy trend is upward.

        Returns:
            bool: True if latest > first accuracy
        """
        if len(self.rounds) < 2:
            return False
        return (
            self.rounds[-1].accuracy >
            self.rounds[0].accuracy
        )

    def save_to_file(
        self,
        output_dir: str = "data/logs"
    ) -> str:
        """
        Save all metrics to JSON file.

        Args:
            output_dir (str): Directory to save

        Returns:
            str: Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(
            output_dir,
            f"metrics_{int(time.time())}.json"
        )

        data = {
            "created_at": self.created_at,
            "total_rounds": len(self.rounds),
            "rounds": self.get_all_metrics()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved: {filepath}")
        return filepath

    def reset(self):
        """Clear all stored metrics."""
        self.rounds = []
        print("MetricsStore reset")