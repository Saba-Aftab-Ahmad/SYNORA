"""
SBFLT-18 Subtask 3: Global Model Management
Handles global model updates and distribution
to browser clients after FedAvg aggregation.

Author: Kashaf Kamran
Sprint: 4
"""

import numpy as np
import json
import time
from typing import List, Optional


class GlobalModel:
    """
    Manages the global model state on the server.
    Stores weights, tracks version history, and
    handles distribution to browser clients.
    """

    def __init__(self, model_architecture: dict):
        """
        Initialise the global model.

        Args:
            model_architecture (dict): Model config
                                      describing layer
                                      shapes and types
        """
        self.architecture = model_architecture
        self.weights = None
        self.version = 0
        self.history = []
        self.created_at = time.time()
        self.last_updated = None

        print("GlobalModel initialised")

    def initialise_weights(
        self,
        initial_weights: List[np.ndarray]
    ):
        """
        Set initial model weights before training.

        Args:
            initial_weights: Initial weight tensors
        """
        self.weights = [
            w.copy() for w in initial_weights
        ]
        self.version = 0
        self.last_updated = time.time()

        print(
            f"Global model weights initialised: "
            f"{len(self.weights)} layers"
        )

    def update_weights(
        self,
        new_weights: List[np.ndarray],
        round_number: int,
        metrics: Optional[dict] = None
    ):
        """
        Update global model weights after FedAvg.

        Args:
            new_weights: Aggregated weight tensors
            round_number (int): Current round number
            metrics (dict): Optional accuracy/loss
        """
        # Store previous weights in history
        if self.weights is not None:
            checkpoint = {
                "version": self.version,
                "round": round_number,
                "timestamp": time.time(),
                "metrics": metrics,
                "weights_shape": [
                    w.shape for w in self.weights
                ]
            }
            self.history.append(checkpoint)

        # Update to new weights
        self.weights = [
            w.copy() for w in new_weights
        ]
        self.version += 1
        self.last_updated = time.time()

        print(
            f"Global model updated: "
            f"version {self.version}, "
            f"round {round_number}"
        )

        if metrics:
            accuracy = metrics.get("accuracy")
            loss = metrics.get("loss")
            if accuracy:
                print(
                    f"  Accuracy: {accuracy:.4f}"
                )
            if loss:
                print(
                    f"  Loss: {loss:.4f}"
                )

    def get_weights(self) -> Optional[List[np.ndarray]]:
        """
        Get current global model weights.

        Returns:
            list: Current weight tensors or None
        """
        if self.weights is None:
            return None
        return [w.copy() for w in self.weights]

    def serialize_weights(self) -> dict:
        """
        Serialize weights for transmission to clients.
        Converts numpy arrays to lists for JSON.

        Returns:
            dict: Serialized weights payload
        """
        if self.weights is None:
            raise ValueError(
                "No weights to serialize"
            )

        serialized = {
            "version": self.version,
            "timestamp": time.time(),
            "num_layers": len(self.weights),
            "weights": [
                {
                    "layer_index": i,
                    "shape": list(w.shape),
                    "data": w.flatten().tolist()
                }
                for i, w in enumerate(self.weights)
            ]
        }

        return serialized

    def deserialize_weights(
        self,
        payload: dict
    ) -> List[np.ndarray]:
        """
        Deserialize weight payload from client.

        Args:
            payload (dict): Serialized weight payload

        Returns:
            list: Deserialized numpy weight arrays
        """
        weights = []
        for layer_data in payload["weights"]:
            shape = tuple(layer_data["shape"])
            data = np.array(
                layer_data["data"]
            ).reshape(shape)
            weights.append(data)
        return weights

    def get_model_info(self) -> dict:
        """
        Get current model information summary.

        Returns:
            dict: Model version and history info
        """
        return {
            "version": self.version,
            "num_layers": (
                len(self.weights)
                if self.weights else 0
            ),
            "last_updated": self.last_updated,
            "history_length": len(self.history),
            "architecture": self.architecture
        }

    def save_checkpoint(
        self,
        round_number: int,
        output_dir: str = "data/checkpoints"
    ):
        """
        Save model checkpoint to file.

        Args:
            round_number (int): Current round
            output_dir (str): Directory to save
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            output_dir,
            f"global_model_round_{round_number}.json"
        )

        payload = self.serialize_weights()
        payload["round_number"] = round_number

        with open(checkpoint_path, "w") as f:
            json.dump(payload, f)

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint file.

        Args:
            checkpoint_path (str): Path to checkpoint
        """
        with open(checkpoint_path, "r") as f:
            payload = json.load(f)

        self.weights = self.deserialize_weights(payload)
        self.version = payload.get("version", 0)

        print(
            f"Checkpoint loaded: {checkpoint_path}"
        )