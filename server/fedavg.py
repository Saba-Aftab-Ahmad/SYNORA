"""
SBFLT-18 Subtask 1: FedAvg Aggregation Algorithm
Implements Federated Averaging as defined in:
McMahan et al. "Communication-Efficient Learning
of Deep Networks from Decentralized Data" (2017)

Author: Kashaf Kamran
Sprint: 4
"""

import numpy as np
from typing import Dict, List, Tuple


def fedavg_aggregate(
    client_weights: Dict[str, List[np.ndarray]], client_sizes: Dict[str, int]
) -> List[np.ndarray]:
    """
    Aggregate client model weights using FedAvg.

    Computes dataset-size weighted average of all
    client weight updates as per McMahan et al.:

        w_global = sum(n_k / n_total * w_k)

    where:
        n_k = number of samples on client k
        n_total = total samples across all clients
        w_k = weight update from client k

    Args:
        client_weights (dict): Maps client_id to list
                               of weight arrays
        client_sizes (dict): Maps client_id to number
                             of local training samples

    Returns:
        list: Aggregated global model weights

    Raises:
        ValueError: If no client weights provided
        ValueError: If weight shapes do not match
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    if not client_sizes:
        raise ValueError("No client sizes provided for aggregation")

    # Validate all clients have matching weight shapes
    client_ids = list(client_weights.keys())
    reference_weights = client_weights[client_ids[0]]
    num_layers = len(reference_weights)

    for client_id in client_ids[1:]:
        weights = client_weights[client_id]
        if len(weights) != num_layers:
            raise ValueError(
                f"Weight layer count mismatch: "
                f"client {client_id} has {len(weights)} "
                f"layers, expected {num_layers}"
            )
        for layer_idx, (w_ref, w_client) in enumerate(zip(reference_weights, weights)):
            if w_ref.shape != w_client.shape:
                raise ValueError(
                    f"Weight shape mismatch at layer "
                    f"{layer_idx}: "
                    f"expected {w_ref.shape}, "
                    f"got {w_client.shape}"
                )

    # Calculate total dataset size
    total_samples = sum(client_sizes[cid] for cid in client_ids if cid in client_sizes)

    if total_samples == 0:
        raise ValueError("Total sample count is zero")

    print(f"\nFedAvg Aggregation:")
    print(f"  Clients: {len(client_ids)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Layers to aggregate: {num_layers}")

    # Compute weighted average per layer
    aggregated_weights = []

    for layer_idx in range(num_layers):
        # Initialise layer accumulator with zeros
        layer_shape = reference_weights[layer_idx].shape
        weighted_sum = np.zeros(layer_shape)

        for client_id in client_ids:
            # Compute this client's contribution weight
            n_k = client_sizes.get(client_id, 0)
            contribution = n_k / total_samples

            # Add weighted client layer weights
            client_layer = client_weights[client_id][layer_idx]
            weighted_sum += contribution * client_layer

            print(
                f"  Layer {layer_idx} | "
                f"Client {client_id}: "
                f"contribution = {contribution:.4f}"
            )

        aggregated_weights.append(weighted_sum)

    print(f"\nAggregation complete: " f"{len(aggregated_weights)} layers aggregated")

    return aggregated_weights


def validate_global_model(
    global_weights: List[np.ndarray], validation_data: List[Tuple], num_classes: int = 3
) -> dict:
    """
    Validate global model on held-out validation set.
    Computes accuracy and loss after aggregation.

    Args:
        global_weights: Aggregated global weights
        validation_data: List of (text_idx, label)
                         validation samples
        num_classes (int): Number of output classes

    Returns:
        dict: Validation metrics (accuracy, loss)
    """
    if not validation_data:
        return {
            "accuracy": None,
            "loss": None,
            "num_samples": 0,
            "error": "No validation data provided",
        }

    # Simulate forward pass validation
    # In production this would use actual TF.js weights
    total_samples = len(validation_data)
    correct = 0
    total_loss = 0.0

    for text_idx, true_label in validation_data:
        # Simulate prediction using weight magnitudes
        # as proxy for confidence
        # In production: actual model inference
        layer_magnitudes = [float(np.mean(np.abs(w))) for w in global_weights[:2]]
        predicted_class = int(sum(layer_magnitudes) * 10) % num_classes

        # Track correctness
        if predicted_class == true_label:
            correct += 1

        # Compute cross-entropy loss
        epsilon = 1e-7
        probs = np.ones(num_classes) * (epsilon / (num_classes - 1))
        probs[predicted_class] = 1 - epsilon
        loss = -np.log(probs[true_label] + epsilon)
        total_loss += loss

    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples

    metrics = {
        "accuracy": round(accuracy, 4),
        "loss": round(avg_loss, 4),
        "num_samples": total_samples,
        "correct": correct,
    }

    print(
        f"\nValidation Results:"
        f"\n  Samples: {total_samples}"
        f"\n  Correct: {correct}"
        f"\n  Accuracy: {accuracy:.4f}"
        f"\n  Loss: {avg_loss:.4f}"
    )

    return metrics


def check_convergence(
    metrics_history: List[dict], patience: int = 3, min_delta: float = 0.001
) -> dict:
    """
    Check whether global model is converging.
    Uses accuracy trend over recent rounds.

    Args:
        metrics_history (list): List of round metrics
        patience (int): Rounds to check for improvement
        min_delta (float): Minimum improvement threshold

    Returns:
        dict: Convergence status and trend analysis
    """
    if len(metrics_history) < 2:
        return {
            "converged": False,
            "trend": "insufficient_data",
            "rounds_checked": len(metrics_history),
        }

    recent = metrics_history[-patience:]
    accuracies = [m.get("accuracy", 0) for m in recent if m.get("accuracy") is not None]

    if len(accuracies) < 2:
        return {"converged": False, "trend": "insufficient_accuracy_data"}

    # Check if improvement is below threshold
    max_improvement = max(accuracies) - min(accuracies)
    is_converged = max_improvement < min_delta
    trend = "converging" if is_converged else "improving"

    print(
        f"\nConvergence Check:"
        f"\n  Rounds checked: {len(recent)}"
        f"\n  Max improvement: {max_improvement:.4f}"
        f"\n  Threshold: {min_delta}"
        f"\n  Status: {trend}"
    )

    return {
        "converged": is_converged,
        "trend": trend,
        "max_improvement": max_improvement,
        "rounds_checked": len(recent),
        "recent_accuracies": accuracies,
    }


def validate_weight_shapes(client_weights: Dict[str, List[np.ndarray]]) -> bool:
    """
    Validate that all clients have matching weight shapes across all layers.

    Args:
        client_weights (dict): Maps client_id to list of weight arrays

    Returns:
        bool: True if all shapes match

    Raises:
        ValueError: If any shape mismatch is detected
    """
    if not client_weights:
        raise ValueError("No client weights provided")

    client_ids = list(client_weights.keys())
    reference_weights = client_weights[client_ids[0]]
    num_layers = len(reference_weights)

    for client_id in client_ids[1:]:
        weights = client_weights[client_id]
        if len(weights) != num_layers:
            raise ValueError(
                f"Layer count mismatch: client {client_id} has "
                f"{len(weights)} layers, expected {num_layers}"
            )
        for layer_idx, (w_ref, w_client) in enumerate(
            zip(reference_weights, weights)
        ):
            if w_ref.shape != w_client.shape:
                raise ValueError(
                    f"Weight shape mismatch at layer {layer_idx}: "
                    f"expected {w_ref.shape}, got {w_client.shape}"
                )

    return True


def aggregate_single_layer(
    client_weights: Dict[str, List[np.ndarray]],
    client_sizes: Dict[str, int],
    layer_idx: int,
) -> np.ndarray:
    """
    Aggregate a single model layer across all clients using FedAvg.

    Args:
        client_weights (dict): Maps client_id to list of weight arrays
        client_sizes (dict): Maps client_id to local dataset size
        layer_idx (int): Index of the layer to aggregate

    Returns:
        np.ndarray: Aggregated weights for the specified layer
    """
    client_ids = list(client_weights.keys())
    total_samples = sum(
        client_sizes.get(cid, 0) for cid in client_ids
    )

    if total_samples == 0:
        raise ValueError("Total sample count is zero")

    reference_layer = client_weights[client_ids[0]][layer_idx]
    weighted_sum = np.zeros(reference_layer.shape)

    for client_id in client_ids:
        n_k = client_sizes.get(client_id, 0)
        contribution = n_k / total_samples
        weighted_sum += contribution * client_weights[client_id][layer_idx]

    return weighted_sum


def compute_client_contributions(client_sizes: dict) -> dict:
    """
    Compute each client's contribution weight
    for FedAvg aggregation.

    Args:
        client_sizes (dict): Maps client_id to
                             local dataset size

    Returns:
        dict: Maps client_id to contribution weight
              (all weights sum to 1.0)
    """
    total = sum(client_sizes.values())

    if total == 0:
        raise ValueError("Total sample count is zero")

    contributions = {
        client_id: size / total for client_id, size in client_sizes.items()
    }

    # Verify contributions sum to 1.0
    contribution_sum = sum(contributions.values())
    assert abs(contribution_sum - 1.0) < 1e-6, (
        f"Contributions do not sum to 1.0: " f"{contribution_sum}"
    )

    print("\nClient Contribution Weights:")
    for client_id, weight in contributions.items():
        print(f"  {client_id}: {weight:.4f} " f"({client_sizes[client_id]} samples)")

    return contributions
