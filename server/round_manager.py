"""
SBFLT-15: Round Management Module
Manages federated learning training rounds on the server.

Author: Kashaf Kamran
Sprint: 3
"""

import time
import uuid
from enum import Enum


class RoundState(Enum):
    """
    Possible states for a federated training round.
    """
    INITIALISED = "initialised"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class FederatedRound:
    """
    Represents a single federated training round.
    Tracks state, clients, and timing information.
    """

    def __init__(self, round_number, min_clients=2):
        self.round_id = str(uuid.uuid4())[:8]
        self.round_number = round_number
        self.min_clients = min_clients
        self.state = RoundState.INITIALISED
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.registered_clients = []
        self.submitted_clients = []
        self.client_updates = {}
        self.global_accuracy = None
        self.global_loss = None

        print(
            f"Round {round_number} initialised "
            f"(ID: {self.round_id})"
        )

    def get_state(self):
        """Return current round state."""
        return self.state

    def set_state(self, new_state):
        """Transition round to a new state."""
        old_state = self.state
        self.state = new_state
        print(
            f"Round {self.round_number}: "
            f"{old_state.value} → {new_state.value}"
        )

    def is_complete(self):
        """Check if round is completed."""
        return self.state == RoundState.COMPLETED

    def is_failed(self):
        """Check if round has failed."""
        return self.state == RoundState.FAILED

    def get_summary(self):
        """Return a summary dictionary for the round."""
        return {
            "round_id": self.round_id,
            "round_number": self.round_number,
            "state": self.state.value,
            "min_clients": self.min_clients,
            "registered_clients": len(
                self.registered_clients
            ),
            "submitted_clients": len(
                self.submitted_clients
            ),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss
        }


class RoundManager:
    """
    Manages all federated training rounds sequentially.
    Coordinates client participation and update collection.
    """

    def __init__(self, min_clients=2, max_rounds=10):
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.current_round = None
        self.round_number = 0
        self.all_rounds = []
        self.registered_clients = {}
        self.is_training = False

        print(
            f"RoundManager initialised: "
            f"min_clients={min_clients}, "
            f"max_rounds={max_rounds}"
        )

    def register_client(self, client_id):
        """Register a browser client with the server."""
        if client_id in self.registered_clients:
            print(
                f"Client already registered: {client_id}"
            )
            return {
                "success": False,
                "message": "Client already registered",
                "client_id": client_id
            }

        self.registered_clients[client_id] = {
            "client_id": client_id,
            "registered_at": time.time(),
            "rounds_participated": 0
        }

        print(
            f"Client registered: {client_id} "
            f"(Total: {len(self.registered_clients)})"
        )

        return {
            "success": True,
            "message": "Client registered successfully",
            "client_id": client_id,
            "total_clients": len(self.registered_clients)
        }

    def start_round(self):
        """Initialise and start a new training round."""
        if self.round_number >= self.max_rounds:
            raise ValueError(
                f"Maximum rounds reached: {self.max_rounds}"
            )

        if len(self.registered_clients) < self.min_clients:
            raise ValueError(
                f"Insufficient clients: "
                f"{len(self.registered_clients)} registered, "
                f"{self.min_clients} required"
            )

        self.round_number += 1
        self.current_round = FederatedRound(
            round_number=self.round_number,
            min_clients=self.min_clients
        )

        for client_id in self.registered_clients:
            self.current_round.registered_clients.append(
                client_id
            )

        self.current_round.set_state(RoundState.COLLECTING)
        self.all_rounds.append(self.current_round)
        self.is_training = True

        print(
            f"\nRound {self.round_number} started. "
            f"Waiting for {self.min_clients} client updates."
        )

        return self.current_round

    def submit_client_update(self, client_id, weights):
        """Accept a model weight update from a client."""
        if self.current_round is None:
            return {
                "success": False,
                "message": "No active round"
            }

        if self.current_round.state != RoundState.COLLECTING:
            return {
                "success": False,
                "message": (
                    f"Round not in COLLECTING state: "
                    f"{self.current_round.state.value}"
                )
            }

        if client_id not in self.registered_clients:
            return {
                "success": False,
                "message": f"Unregistered client: {client_id}"
            }

        if client_id in self.current_round.submitted_clients:
            return {
                "success": False,
                "message": "Client already submitted this round"
            }

        self.current_round.client_updates[client_id] = weights
        self.current_round.submitted_clients.append(client_id)
        self.registered_clients[client_id][
            "rounds_participated"
        ] += 1

        submitted = len(self.current_round.submitted_clients)
        required = self.current_round.min_clients

        print(
            f"Update received from {client_id} "
            f"({submitted}/{required} received)"
        )

        threshold_met = submitted >= required

        return {
            "success": True,
            "message": "Update received",
            "submitted": submitted,
            "required": required,
            "threshold_met": threshold_met,
            "ready_to_aggregate": threshold_met
        }

    def complete_round(
        self, global_accuracy=None, global_loss=None
    ):
        """Mark the current round as completed."""
        if self.current_round is None:
            return {
                "success": False,
                "message": "No active round to complete"
            }

        self.current_round.global_accuracy = global_accuracy
        self.current_round.global_loss = global_loss
        self.current_round.end_time = time.time()
        self.current_round.duration = (
            self.current_round.end_time -
            self.current_round.start_time
        )

        self.current_round.set_state(RoundState.COMPLETED)

        summary = self.current_round.get_summary()

        print(
            f"\nRound {self.current_round.round_number} "
            f"completed in "
            f"{self.current_round.duration:.2f}s"
        )

        if global_accuracy:
            print(f"Global accuracy: {global_accuracy:.4f}")
        if global_loss:
            print(f"Global loss: {global_loss:.4f}")

        self._notify_round_complete()

        return {
            "success": True,
            "message": "Round completed",
            "round_summary": summary
        }

    def fail_round(self, reason="Unknown error"):
        """Mark the current round as failed."""
        if self.current_round is None:
            return

        self.current_round.set_state(RoundState.FAILED)
        self.current_round.end_time = time.time()

        print(
            f"Round {self.current_round.round_number} "
            f"FAILED: {reason}"
        )

    def _notify_round_complete(self):
        """Log round completion notification."""
        print(
            f"\nNotifying {len(self.registered_clients)} "
            f"clients: Round "
            f"{self.current_round.round_number} complete"
        )
        for client_id in self.registered_clients:
            print(f"  → Client {client_id} notified")

    def get_current_round_status(self):
        """Get current round status summary."""
        if self.current_round is None:
            return {
                "active_round": False,
                "round_number": 0,
                "total_rounds_completed": len(self.all_rounds)
            }

        return {
            "active_round": True,
            "round_summary": self.current_round.get_summary(),
            "total_rounds_completed": len([
                r for r in self.all_rounds
                if r.is_complete()
            ])
        }

    def get_all_client_updates(self):
        """Get all submitted client updates."""
        if self.current_round is None:
            return {}
        return self.current_round.client_updates

    def get_round_log(self):
        """Get complete log of all rounds."""
        return [
            round_obj.get_summary()
            for round_obj in self.all_rounds
        ]

    def get_velocity_stats(self):
        """Calculate round velocity statistics."""
        completed_rounds = [
            r for r in self.all_rounds
            if r.is_complete()
        ]

        if not completed_rounds:
            return {
                "completed_rounds": 0,
                "avg_duration": None,
                "avg_accuracy": None,
                "avg_loss": None
            }

        durations = [
            r.duration for r in completed_rounds
            if r.duration
        ]
        accuracies = [
            r.global_accuracy for r in completed_rounds
            if r.global_accuracy is not None
        ]
        losses = [
            r.global_loss for r in completed_rounds
            if r.global_loss is not None
        ]

        return {
            "completed_rounds": len(completed_rounds),
            "avg_duration": (
                sum(durations) / len(durations)
                if durations else None
            ),
            "avg_accuracy": (
                sum(accuracies) / len(accuracies)
                if accuracies else None
            ),
            "avg_loss": (
                sum(losses) / len(losses)
                if losses else None
            )
        }

    def reset(self):
        """Reset the round manager to initial state."""
        self.current_round = None
        self.round_number = 0
        self.all_rounds = []
        self.registered_clients = {}
        self.is_training = False
        print("RoundManager reset to initial state")