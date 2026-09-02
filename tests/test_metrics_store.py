"""
Unit Tests for SBFLT-23: Convergence Metrics Store
Tests metrics collection, series extraction,
chart data formatting, and file persistence.
"""

import sys
import os
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.metrics_store import MetricsStore, RoundMetrics

TEST_OUTPUT_DIR = "data/test_logs"


def test_add_round_metrics():
    """Test adding metrics for completed rounds"""
    print("\nTest 1: Add round metrics")

    store = MetricsStore()
    store.add_round_metrics(round_number=1, accuracy=0.65, loss=0.85, num_clients=3)

    assert len(store.rounds) == 1
    assert store.rounds[0].accuracy == 0.65
    assert store.rounds[0].loss == 0.85
    assert store.rounds[0].round_number == 1

    print("✅ PASS: Round metrics added correctly")


def test_multiple_rounds():
    """Test adding metrics for multiple rounds"""
    print("\nTest 2: Multiple rounds")

    store = MetricsStore()
    for i in range(1, 6):
        store.add_round_metrics(
            round_number=i, accuracy=0.5 + i * 0.05, loss=1.0 - i * 0.08, num_clients=3
        )

    assert len(store.rounds) == 5
    print(f"✅ PASS: {len(store.rounds)} rounds stored")


def test_accuracy_series():
    """Test accuracy series extraction"""
    print("\nTest 3: Accuracy series")

    store = MetricsStore()
    store.add_round_metrics(1, 0.60, 0.90, 3)
    store.add_round_metrics(2, 0.70, 0.75, 3)
    store.add_round_metrics(3, 0.78, 0.62, 3)

    series = store.get_accuracy_series()

    assert len(series) == 3
    assert series[0] == 0.60
    assert series[1] == 0.70
    assert series[2] == 0.78

    print("✅ PASS: Accuracy series correct")


def test_loss_series():
    """Test loss series extraction"""
    print("\nTest 4: Loss series")

    store = MetricsStore()
    store.add_round_metrics(1, 0.60, 0.90, 3)
    store.add_round_metrics(2, 0.70, 0.75, 3)

    series = store.get_loss_series()

    assert len(series) == 2
    assert series[0] == 0.90
    assert series[1] == 0.75

    print("✅ PASS: Loss series correct")


def test_round_numbers_series():
    """Test round numbers for x-axis"""
    print("\nTest 5: Round numbers series")

    store = MetricsStore()
    store.add_round_metrics(1, 0.60, 0.90, 3)
    store.add_round_metrics(2, 0.70, 0.75, 3)
    store.add_round_metrics(3, 0.78, 0.62, 3)

    labels = store.get_round_numbers()

    assert labels == [1, 2, 3]
    print("✅ PASS: Round numbers correct")


def test_chart_data_format():
    """Test chart data matches Chart.js format"""
    print("\nTest 6: Chart data format")

    store = MetricsStore()
    store.add_round_metrics(1, 0.60, 0.90, 3)
    store.add_round_metrics(2, 0.70, 0.75, 3)

    data = store.get_chart_data()

    assert "labels" in data
    assert "accuracy" in data
    assert "loss" in data
    assert "num_rounds" in data
    assert data["num_rounds"] == 2
    assert len(data["labels"]) == 2
    assert len(data["accuracy"]) == 2
    assert len(data["loss"]) == 2

    print("✅ PASS: Chart data format correct")


def test_latest_metrics():
    """Test getting latest round metrics"""
    print("\nTest 7: Latest metrics")

    store = MetricsStore()
    store.add_round_metrics(1, 0.60, 0.90, 3)
    store.add_round_metrics(2, 0.75, 0.65, 3)

    latest = store.get_latest_metrics()

    assert latest is not None
    assert latest["round_number"] == 2
    assert latest["accuracy"] == 0.75
    assert latest["loss"] == 0.65

    print("✅ PASS: Latest metrics correct")


def test_empty_store_latest_metrics():
    """Test latest metrics on empty store"""
    print("\nTest 8: Empty store latest metrics")

    store = MetricsStore()
    latest = store.get_latest_metrics()

    assert latest is None
    print("✅ PASS: Empty store returns None")


def test_accuracy_improving():
    """Test accuracy improvement detection"""
    print("\nTest 9: Accuracy improving detection")

    store = MetricsStore()
    store.add_round_metrics(1, 0.55, 0.90, 3)
    store.add_round_metrics(2, 0.65, 0.75, 3)
    store.add_round_metrics(3, 0.74, 0.61, 3)

    assert store.is_accuracy_improving() == True
    print("✅ PASS: Accuracy improving detected")

    # Test declining accuracy
    store2 = MetricsStore()
    store2.add_round_metrics(1, 0.80, 0.40, 3)
    store2.add_round_metrics(2, 0.72, 0.55, 3)

    assert store2.is_accuracy_improving() == False
    print("✅ PASS: Declining accuracy detected")


def test_invalid_accuracy_rejected():
    """Test invalid accuracy value rejected"""
    print("\nTest 10: Invalid accuracy rejected")

    store = MetricsStore()
    try:
        store.add_round_metrics(1, 1.5, 0.5, 3)
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✅ PASS: ValueError raised: {e}")


def test_negative_loss_rejected():
    """Test negative loss value rejected"""
    print("\nTest 11: Negative loss rejected")

    store = MetricsStore()
    try:
        store.add_round_metrics(1, 0.6, -0.5, 3)
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✅ PASS: ValueError raised: {e}")


def test_save_metrics_to_file():
    """Test saving metrics to JSON file"""
    print("\nTest 12: Save metrics to file")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    try:
        store = MetricsStore()
        store.add_round_metrics(1, 0.60, 0.90, 3)
        store.add_round_metrics(2, 0.70, 0.75, 3)

        filepath = store.save_to_file(TEST_OUTPUT_DIR)

        assert os.path.exists(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert data["total_rounds"] == 2
        assert len(data["rounds"]) == 2

        print("✅ PASS: Metrics saved and verified")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    finally:
        shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)


def test_reset_store():
    """Test resetting metrics store"""
    print("\nTest 13: Reset store")

    store = MetricsStore()
    store.add_round_metrics(1, 0.60, 0.90, 3)
    store.add_round_metrics(2, 0.70, 0.75, 3)

    assert len(store.rounds) == 2
    store.reset()
    assert len(store.rounds) == 0

    print("✅ PASS: Store reset correctly")


if __name__ == "__main__":
    print("=" * 55)
    print("Running SBFLT-23 Metrics Store Tests")
    print("=" * 55)

    test_add_round_metrics()
    test_multiple_rounds()
    test_accuracy_series()
    test_loss_series()
    test_round_numbers_series()
    test_chart_data_format()
    test_latest_metrics()
    test_empty_store_latest_metrics()
    test_accuracy_improving()
    test_invalid_accuracy_rejected()
    test_negative_loss_rejected()
    test_save_metrics_to_file()
    test_reset_store()

    print("\n" + "=" * 55)
    print("All SBFLT-23 tests completed")
    print("=" * 55)
