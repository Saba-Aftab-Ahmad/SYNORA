"""
Unit Tests for SBFLT-15: Round Management
Tests round initialisation, client registration,
update collection, and completion logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from server.round_manager import (
    RoundManager,
    FederatedRound,
    RoundState
)


def test_round_initialisation():
    """Test round initialises with correct state"""
    print("\nTest 1: Round initialisation")

    round_obj = FederatedRound(
        round_number=1,
        min_clients=2
    )

    assert round_obj.round_number == 1
    assert round_obj.state == RoundState.INITIALISED
    assert round_obj.min_clients == 2
    assert len(round_obj.submitted_clients) == 0
    assert len(round_obj.client_updates) == 0

    print("✅ PASS: Round initialised correctly")


def test_client_registration():
    """Test client registration and duplicate rejection"""
    print("\nTest 2: Client registration")

    manager = RoundManager(min_clients=2)

    result1 = manager.register_client("client_001")
    assert result1["success"] == True
    assert result1["total_clients"] == 1

    result2 = manager.register_client("client_002")
    assert result2["success"] == True
    assert result2["total_clients"] == 2

    # Duplicate registration should fail
    result3 = manager.register_client("client_001")
    assert result3["success"] == False

    print(
        "✅ PASS: Client registration and "
        "duplicate rejection working"
    )


def test_start_round():
    """Test round starts correctly with clients"""
    print("\nTest 3: Start round")

    manager = RoundManager(min_clients=2)
    manager.register_client("client_001")
    manager.register_client("client_002")

    round_obj = manager.start_round()

    assert round_obj is not None
    assert round_obj.round_number == 1
    assert round_obj.state == RoundState.COLLECTING
    assert len(round_obj.registered_clients) == 2

    print("✅ PASS: Round started correctly")


def test_insufficient_clients():
    """Test round fails without minimum clients"""
    print("\nTest 4: Insufficient clients check")

    manager = RoundManager(min_clients=3)
    manager.register_client("client_001")
    manager.register_client("client_002")

    try:
        manager.start_round()
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(
            f"✅ PASS: Correctly raised ValueError: {e}"
        )


def test_client_update_submission():
    """Test client update submission and threshold"""
    print("\nTest 5: Client update submission")

    manager = RoundManager(min_clients=2)
    manager.register_client("client_001")
    manager.register_client("client_002")
    manager.start_round()

    # Submit first update
    dummy_weights = [0.1, 0.2, 0.3]
    result1 = manager.submit_client_update(
        "client_001",
        dummy_weights
    )
    assert result1["success"] == True
    assert result1["threshold_met"] == False
    assert result1["submitted"] == 1

    # Submit second update - should meet threshold
    result2 = manager.submit_client_update(
        "client_002",
        dummy_weights
    )
    assert result2["success"] == True
    assert result2["threshold_met"] == True
    assert result2["submitted"] == 2

    print(
        "✅ PASS: Client updates and "
        "threshold detection working"
    )


def test_duplicate_submission_rejected():
    """Test duplicate client submission is rejected"""
    print("\nTest 6: Duplicate submission rejected")

    manager = RoundManager(min_clients=2)
    manager.register_client("client_001")
    manager.register_client("client_002")
    manager.start_round()

    dummy_weights = [0.1, 0.2, 0.3]
    manager.submit_client_update(
        "client_001",
        dummy_weights
    )

    # Duplicate submission
    result = manager.submit_client_update(
        "client_001",
        dummy_weights
    )
    assert result["success"] == False

    print(
        "✅ PASS: Duplicate submission "
        "correctly rejected"
    )


def test_complete_round():
    """Test round completion with metrics"""
    print("\nTest 7: Round completion")

    manager = RoundManager(min_clients=2)
    manager.register_client("client_001")
    manager.register_client("client_002")
    manager.start_round()

    dummy_weights = [0.1, 0.2, 0.3]
    manager.submit_client_update(
        "client_001",
        dummy_weights
    )
    manager.submit_client_update(
        "client_002",
        dummy_weights
    )

    result = manager.complete_round(
        global_accuracy=0.85,
        global_loss=0.32
    )

    assert result["success"] == True
    assert manager.current_round.state == RoundState.COMPLETED
    assert manager.current_round.global_accuracy == 0.85
    assert manager.current_round.global_loss == 0.32
    assert manager.current_round.duration is not None

    print("✅ PASS: Round completed with metrics")


def test_round_log():
    """Test complete round log generation"""
    print("\nTest 8: Round log generation")

    manager = RoundManager(min_clients=2)
    manager.register_client("client_001")
    manager.register_client("client_002")

    # Complete 2 rounds
    for i in range(2):
        manager.start_round()
        manager.submit_client_update(
            "client_001", [0.1]
        )
        manager.submit_client_update(
            "client_002", [0.2]
        )
        manager.complete_round(
            global_accuracy=0.8 + i * 0.05,
            global_loss=0.4 - i * 0.05
        )

    log = manager.get_round_log()
    assert len(log) == 2
    assert log[0]["round_number"] == 1
    assert log[1]["round_number"] == 2

    print(
        f"✅ PASS: Round log contains "
        f"{len(log)} entries"
    )


def test_velocity_stats():
    """Test velocity statistics calculation"""
    print("\nTest 9: Velocity statistics")

    manager = RoundManager(min_clients=2)
    manager.register_client("client_001")
    manager.register_client("client_002")
    manager.start_round()
    manager.submit_client_update("client_001", [0.1])
    manager.submit_client_update("client_002", [0.2])
    manager.complete_round(
        global_accuracy=0.85,
        global_loss=0.32
    )

    stats = manager.get_velocity_stats()
    assert stats["completed_rounds"] == 1
    assert stats["avg_accuracy"] == 0.85
    assert stats["avg_loss"] == 0.32

    print("✅ PASS: Velocity statistics correct")


def test_round_state_transitions():
    """Test round state transitions are logged"""
    print("\nTest 10: Round state transitions")

    round_obj = FederatedRound(
        round_number=1,
        min_clients=2
    )

    assert round_obj.state == RoundState.INITIALISED
    round_obj.set_state(RoundState.COLLECTING)
    assert round_obj.state == RoundState.COLLECTING
    round_obj.set_state(RoundState.AGGREGATING)
    assert round_obj.state == RoundState.AGGREGATING
    round_obj.set_state(RoundState.COMPLETED)
    assert round_obj.state == RoundState.COMPLETED
    assert round_obj.is_complete() == True

    print(
        "✅ PASS: All state transitions "
        "working correctly"
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Running SBFLT-15 Round Manager Tests")
    print("=" * 50)

    test_round_initialisation()
    test_client_registration()
    test_start_round()
    test_insufficient_clients()
    test_client_update_submission()
    test_duplicate_submission_rejected()
    test_complete_round()
    test_round_log()
    test_velocity_stats()
    test_round_state_transitions()

    print("\n" + "=" * 50)
    print("All SBFLT-15 tests completed")
    print("=" * 50)