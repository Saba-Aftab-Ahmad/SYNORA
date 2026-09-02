"""
US-13: Experiment Results Tracker
Tracks per-round accuracy, loss, and client participation data.
"""
import json
import csv
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name: str = "fl_experiment"):
        self.experiment_name = experiment_name
        self.start_time = datetime.now().isoformat()
        self.config = {}
        self.rounds = []

    def set_config(self, config: dict):
        """Store experiment configuration parameters."""
        self.config = config
        print(f"[CONFIG] Experiment config set: {config}")

    def log_round(self, round_num: int, accuracy: float,
                  loss: float, participating_clients: list):
        """Log results for a single round."""
        round_data = {
            "round": round_num,
            "accuracy": round(accuracy, 4),
            "loss": round(loss, 4),
            "participating_clients": participating_clients,
            "client_count": len(participating_clients),
            "timestamp": datetime.now().isoformat()
        }
        self.rounds.append(round_data)
        print(f"[ROUND {round_num}] "
              f"Accuracy: {accuracy:.4f} | "
              f"Loss: {loss:.4f} | "
              f"Clients: {len(participating_clients)}")
        return round_data

    def export_json(self, filepath: str = None) -> str:
        """Export results to JSON format."""
        if filepath is None:
            filepath = f"{self.experiment_name}_results.json"

        export_data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "export_time": datetime.now().isoformat(),
            "configuration": self.config,
            "total_rounds": len(self.rounds),
            "rounds": self.rounds
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"[EXPORT] JSON saved: {filepath}")
        return filepath

    def export_csv(self, filepath: str = None) -> str:
        """Export results to CSV format."""
        if filepath is None:
            filepath = f"{self.experiment_name}_results.csv"

        if not self.rounds:
            raise ValueError("No rounds to export")

        fieldnames = [
            "round",
            "accuracy",
            "loss",
            "client_count",
            "participating_clients",
            "timestamp",
            "experiment_name",
            "min_clients",
            "selection_count",
            "max_rounds"
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for round_data in self.rounds:
                row = {
                    "round": round_data["round"],
                    "accuracy": round_data["accuracy"],
                    "loss": round_data["loss"],
                    "client_count": round_data["client_count"],
                    "participating_clients": ",".join(
                        round_data["participating_clients"]
                    ),
                    "timestamp": round_data["timestamp"],
                    "experiment_name": self.experiment_name,
                    "min_clients": self.config.get("min_clients", ""),
                    "selection_count": self.config.get("selection_count", ""),
                    "max_rounds": self.config.get("max_rounds", "")
                }
                writer.writerow(row)

        print(f"[EXPORT] CSV saved: {filepath}")
        return filepath

    def get_summary(self) -> dict:
        """Return experiment summary statistics."""
        if not self.rounds:
            return {"message": "No rounds completed yet"}

        accuracies = [r["accuracy"] for r in self.rounds]
        losses = [r["loss"] for r in self.rounds]

        return {
            "experiment_name": self.experiment_name,
            "total_rounds": len(self.rounds),
            "best_accuracy": max(accuracies),
            "final_accuracy": accuracies[-1],
            "best_loss": min(losses),
            "final_loss": losses[-1],
            "avg_accuracy": round(sum(accuracies)/len(accuracies), 4),
            "avg_loss": round(sum(losses)/len(losses), 4),
            "avg_clients_per_round": round(
                sum(r["client_count"] for r in self.rounds) / len(self.rounds), 2
            )
        }