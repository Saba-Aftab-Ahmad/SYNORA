"""
US-12: Random Client Selection Strategy
Randomly selects a subset of clients per federated learning round.
"""

import random
from collections import defaultdict

class ClientSelector:
    def __init__(self, selection_count: int = 2):
        """
        Initialize the client selector.
        
        Args:
            selection_count: Number of clients to select per round
        """
        self.selection_count = selection_count
        self.selection_history = defaultdict(int)
        self.round_number = 0

    def set_selection_count(self, count: int):
        """Configure how many clients to select per round."""
        if count < 1:
            raise ValueError("Selection count must be at least 1")
        self.selection_count = count
        print(f"[CONFIG] Selection count set to {count}")

    def select_clients(self, available_clients: list) -> list:
        """
        Randomly select subset of clients for this round.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            List of randomly selected client IDs
        """
        if len(available_clients) < self.selection_count:
            print(f"[WARNING] Not enough clients. "
                  f"Available: {len(available_clients)}, "
                  f"Required: {self.selection_count}")
            return available_clients

        # Random selection
        selected = random.sample(available_clients, self.selection_count)
        
        # Track selection history
        self.round_number += 1
        for client in selected:
            self.selection_history[client] += 1

        print(f"[ROUND {self.round_number}] "
              f"Selected {len(selected)}/{len(available_clients)} clients: "
              f"{selected}")
        
        return selected

    def get_selection_history(self) -> dict:
        """Return how many times each client was selected."""
        return dict(self.selection_history)

    def get_frequency_distribution(self, available_clients: list) -> dict:
        """
        Calculate selection frequency for each client.
        Shows if selection is approximately uniform.
        """
        total_selections = sum(self.selection_history.values())
        if total_selections == 0:
            return {}

        distribution = {}
        for client in available_clients:
            count = self.selection_history.get(client, 0)
            percentage = (count / total_selections) * 100
            distribution[client] = {
                "selected_count": count,
                "percentage": round(percentage, 2)
            }
        return distribution

    def check_bias(self, available_clients: list) -> dict:
        """
        Check if any client is consistently biased.
        Uniform distribution expected = 100/num_clients %
        Bias threshold = 2x expected frequency
        """
        if self.round_number == 0:
            return {"bias_detected": False, "message": "No rounds completed yet"}

        expected_pct = (self.selection_count / len(available_clients)) * 100
        bias_threshold = expected_pct * 2

        biased_clients = []
        for client in available_clients:
            count = self.selection_history.get(client, 0)
            actual_pct = (count / self.round_number) * 100
            if actual_pct > bias_threshold:
                biased_clients.append(client)

        return {
            "bias_detected": len(biased_clients) > 0,
            "biased_clients": biased_clients,
            "expected_percentage": round(expected_pct, 2),
            "rounds_completed": self.round_number
        }

    def get_stats(self) -> dict:
        """Return full selection statistics."""
        return {
            "selection_count": self.selection_count,
            "rounds_completed": self.round_number,
            "selection_history": self.get_selection_history()
        }