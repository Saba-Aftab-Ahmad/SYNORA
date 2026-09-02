"""
US-11: Aggregation Participation Threshold Configuration
Manages minimum client threshold before aggregation proceeds.
"""

# Default configuration
DEFAULT_CONFIG = {
    "min_clients": 2,
    "max_rounds": 10,
    "current_round": 0
}

class AggregationConfig:
    def __init__(self, min_clients=2, max_rounds=10):
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.current_round = 0

    def set_threshold(self, min_clients: int):
        """Set minimum clients required before aggregation."""
        if min_clients < 1:
            raise ValueError("Minimum clients must be at least 1")
        self.min_clients = min_clients
        print(f"[CONFIG] Threshold set to {min_clients} clients")

    def can_aggregate(self, connected_clients: int) -> bool:
        """Check if enough clients are connected to aggregate."""
        result = connected_clients >= self.min_clients
        status = "READY" if result else "WAITING"
        print(f"[ROUND {self.current_round}] Status: {status} | "
              f"Connected: {connected_clients} | "
              f"Required: {self.min_clients}")
        return result

    def next_round(self):
        """Move to next round."""
        self.current_round += 1

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            "min_clients": self.min_clients,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round
        }