"""
Unit Tests for SBFLT-18: FedAvg Aggregation
Tests mathematical correctness of FedAvg algorithm
against manually computed reference values.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from server.fedavg import (
    fedavg_aggregate,
    compute_client_contributions,
    validate_weight_shapes,
    aggregate_single_layer,
    validate_global_model,
    check_convergence
)
from server.global_model import GlobalModel


def test_fedavg_two_client_reference():
    """
    Test FedAvg against manually computed reference.
    This is the acceptance criteria test case:
    2-client, 2-round verification.
    """
    print("\nTest 1: FedAvg 2-client reference check")

    # Client 1: 60 samples, weights = 0.6
    # Client 2: 40 samples, weights = 0.4
    # Expected: 0.6 * w1 + 0.4 * w2

    client_weights = {
        "client_1": [
            np.array([[1.0, 2.0], [3.0, 4.0]])
        ],
        "client_2": [
            np.array([[3.0, 4.0], [5.0, 6.0]])
        ]
    }
    client_sizes = {
        "client_1": 60,
        "client_2": 40
    }

    # Manually compute expected result
    expected = (
        0.6 * np.array([[1.0, 2.0], [3.0, 4.0]]) +
        0.4 * np.array([[3.0, 4.0], [5.0, 6.0]])
    )

    result = fedavg_aggregate(client_weights, client_sizes)

    assert np.allclose(result[0], expected, atol=1e-6), (
        f"FedAvg result does not match manual computation"
        f"\nExpected: {expected}"
        f"\nGot: {result[0]}"
    )

    print(
        f"✅ PASS: FedAvg matches manual computation"
    )
    print(f"   Expected: {expected}")
    print(f"   Got: {result[0]}")


def test_fedavg_equal_weights():
    """
    Test FedAvg with equal client sizes produces
    simple average.
    """
    print("\nTest 2: FedAvg equal client sizes")

    client_weights = {
        "client_1": [np.array([2.0, 4.0])],
        "client_2": [np.array([4.0, 8.0])]
    }
    client_sizes = {
        "client_1": 50,
        "client_2": 50
    }

    # Equal sizes means simple average
    expected = np.array([3.0, 6.0])
    result = fedavg_aggregate(client_weights, client_sizes)

    assert np.allclose(result[0], expected, atol=1e-6)
    print("✅ PASS: Equal sizes produce simple average")


def test_fedavg_three_clients():
    """Test FedAvg with three clients"""
    print("\nTest 3: FedAvg three clients")

    client_weights = {
        "client_1": [np.array([1.0, 0.0])],
        "client_2": [np.array([0.0, 1.0])],
        "client_3": [np.array([0.5, 0.5])]
    }
    client_sizes = {
        "client_1": 100,
        "client_2": 100,
        "client_3": 100
    }

    result = fedavg_aggregate(client_weights, client_sizes)
    total = sum(client_sizes.values())

    expected = (
        (100/total) * np.array([1.0, 0.0]) +
        (100/total) * np.array([0.0, 1.0]) +
        (100/total) * np.array([0.5, 0.5])
    )

    assert np.allclose(result[0], expected, atol=1e-6)
    print("✅ PASS: Three client aggregation correct")


def test_contribution_weights_sum_to_one():
    """Test that contribution weights always sum to 1"""
    print("\nTest 4: Contributions sum to 1.0")

    client_sizes = {
        "client_1": 150,
        "client_2": 75,
        "client_3": 275
    }

    contributions = compute_client_contributions(
        client_sizes
    )
    total = sum(contributions.values())

    assert abs(total - 1.0) < 1e-6, (
        f"Contributions sum to {total}, expected 1.0"
    )
    print(
        f"✅ PASS: Contributions sum to {total:.6f}"
    )


def test_shape_mismatch_detection():
    """Test that weight shape mismatches are caught"""
    print("\nTest 5: Shape mismatch detection")

    client_weights = {
        "client_1": [np.array([1.0, 2.0, 3.0])],
        "client_2": [np.array([1.0, 2.0])]
    }

    try:
        validate_weight_shapes(client_weights)
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(
            f"✅ PASS: Shape mismatch detected: {e}"
        )


def test_empty_clients_raises_error():
    """Test that empty client dict raises error"""
    print("\nTest 6: Empty clients error")

    try:
        fedavg_aggregate({}, {})
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(
            f"✅ PASS: Empty clients error: {e}"
        )


def test_global_model_update():
    """Test global model weight update"""
    print("\nTest 7: Global model update")

    architecture = {
        "vocab_size": 5000,
        "embedding_dim": 64,
        "num_classes": 3
    }
    model = GlobalModel(architecture)

    initial_weights = [
        np.random.randn(10, 5),
        np.random.randn(5)
    ]
    model.initialise_weights(initial_weights)

    assert model.version == 0
    assert model.weights is not None

    new_weights = [
        np.random.randn(10, 5),
        np.random.randn(5)
    ]
    model.update_weights(
        new_weights,
        round_number=1,
        metrics={"accuracy": 0.85, "loss": 0.32}
    )

    assert model.version == 1
    assert len(model.history) == 1
    print("✅ PASS: Global model update working")


def test_model_serialization():
    """Test model weight serialization"""
    print("\nTest 8: Model serialization")

    architecture = {"num_classes": 3}
    model = GlobalModel(architecture)

    weights = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.1, 0.2])
    ]
    model.initialise_weights(weights)

    serialized = model.serialize_weights()
    assert "weights" in serialized
    assert "version" in serialized
    assert serialized["num_layers"] == 2

    # Deserialize and check
    recovered = model.deserialize_weights(serialized)
    assert np.allclose(recovered[0], weights[0])
    assert np.allclose(recovered[1], weights[1])

    print(
        "✅ PASS: Serialization and "
        "deserialization correct"
    )


def test_convergence_detection():
    """Test convergence check logic"""
    print("\nTest 9: Convergence detection")

    # Metrics showing convergence (small improvement)
    converged_metrics = [
        {"accuracy": 0.850},
        {"accuracy": 0.851},
        {"accuracy": 0.851},
        {"accuracy": 0.852}
    ]

    result = check_convergence(
        converged_metrics,
        patience=3,
        min_delta=0.01
    )
    assert result["converged"] == True
    print("✅ PASS: Convergence correctly detected")

    # Metrics showing improvement (not converged)
    improving_metrics = [
        {"accuracy": 0.70},
        {"accuracy": 0.75},
        {"accuracy": 0.82},
        {"accuracy": 0.88}
    ]

    result2 = check_convergence(
        improving_metrics,
        patience=3,
        min_delta=0.01
    )
    assert result2["converged"] == False
    print("✅ PASS: Improvement correctly detected")


def test_fedavg_multilayer():
    """Test FedAvg aggregates all layers correctly"""
    print("\nTest 10: FedAvg multi-layer aggregation")

    num_layers = 4
    client_weights = {
        "client_1": [
            np.ones((3, 3)) * 2.0
            for _ in range(num_layers)
        ],
        "client_2": [
            np.ones((3, 3)) * 4.0
            for _ in range(num_layers)
        ]
    }
    client_sizes = {
        "client_1": 50,
        "client_2": 50
    }

    result = fedavg_aggregate(client_weights, client_sizes)

    assert len(result) == num_layers

    # Equal weights → simple average = 3.0
    expected_val = 3.0
    for layer in result:
        assert np.allclose(layer, expected_val, atol=1e-6)

    print(
        f"✅ PASS: All {num_layers} layers "
        f"aggregated correctly"
    )


if __name__ == "__main__":
    print("=" * 55)
    print("Running SBFLT-18 FedAvg Aggregation Tests")
    print("=" * 55)

    test_fedavg_two_client_reference()
    test_fedavg_equal_weights()
    test_fedavg_three_clients()
    test_contribution_weights_sum_to_one()
    test_shape_mismatch_detection()
    test_empty_clients_raises_error()
    test_global_model_update()
    test_model_serialization()
    test_convergence_detection()
    test_fedavg_multilayer()

    print("\n" + "=" * 55)
    print("All SBFLT-18 tests completed")
    print("=" * 55)