"""
Unit Tests for SBFLT-24: Experiment Configuration
Tests schema validation, file loading, and writing.
"""

import sys
import os
import json
import shutil

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from server.experiment_config import (
    ExperimentConfig,
    ConfigValidationError,
    DEFAULT_CONFIG
)
from server.config_manager import ConfigManager


# Test output directory
TEST_CONFIG_DIR = "data/test_configs"


def test_default_config_valid():
    """Test default config passes validation"""
    print("\nTest 1: Default config is valid")
    try:
        config = ExperimentConfig()
        assert config.get("num_rounds") == 10
        assert config.get("learning_rate") == 0.01
        assert config.get("num_clients") == 3
        print("✅ PASS: Default config valid")
    except Exception as e:
        print(f"❌ FAIL: {e}")


def test_custom_config_valid():
    """Test custom config with valid values"""
    print("\nTest 2: Custom config valid")
    custom = {
        "experiment_name": "test_run",
        "num_rounds": 5,
        "num_clients": 4,
        "min_clients_per_round": 2,
        "learning_rate": 0.001,
        "partition_type": "iid"
    }
    try:
        config = ExperimentConfig(custom)
        assert config.get("num_rounds") == 5
        assert config.get("learning_rate") == 0.001
        print("✅ PASS: Custom config valid")
    except Exception as e:
        print(f"❌ FAIL: {e}")


def test_zero_rounds_rejected():
    """Test that zero rounds raises error"""
    print("\nTest 3: Zero rounds rejected")
    try:
        ExperimentConfig({"num_rounds": 0})
        print("❌ FAIL: Should have raised error")
    except ConfigValidationError as e:
        print(f"✅ PASS: {e}")


def test_negative_learning_rate_rejected():
    """Test negative learning rate raises error"""
    print("\nTest 4: Negative learning rate rejected")
    try:
        ExperimentConfig({"learning_rate": -0.01})
        print("❌ FAIL: Should have raised error")
    except ConfigValidationError as e:
        print(f"✅ PASS: {e}")


def test_invalid_partition_type_rejected():
    """Test invalid partition type raises error"""
    print("\nTest 5: Invalid partition type rejected")
    try:
        ExperimentConfig({"partition_type": "random"})
        print("❌ FAIL: Should have raised error")
    except ConfigValidationError as e:
        print(f"✅ PASS: {e}")


def test_invalid_language_rejected():
    """Test invalid language raises error"""
    print("\nTest 6: Invalid language rejected")
    try:
        ExperimentConfig({
            "languages": ["dholuo", "french"]
        })
        print("❌ FAIL: Should have raised error")
    except ConfigValidationError as e:
        print(f"✅ PASS: {e}")


def test_min_clients_exceeds_num_clients():
    """Test min_clients > num_clients raises error"""
    print("\nTest 7: min_clients > num_clients rejected")
    try:
        ExperimentConfig({
            "num_clients": 3,
            "min_clients_per_round": 5
        })
        print("❌ FAIL: Should have raised error")
    except ConfigValidationError as e:
        print(f"✅ PASS: {e}")


def test_learning_rate_exceeds_one():
    """Test learning rate > 1.0 raises error"""
    print("\nTest 8: Learning rate > 1.0 rejected")
    try:
        ExperimentConfig({"learning_rate": 1.5})
        print("❌ FAIL: Should have raised error")
    except ConfigValidationError as e:
        print(f"✅ PASS: {e}")


def test_save_and_load_config():
    """Test saving and loading config round-trip"""
    print("\nTest 9: Save and load config")
    os.makedirs(TEST_CONFIG_DIR, exist_ok=True)

    try:
        manager = ConfigManager(TEST_CONFIG_DIR)
        config = ExperimentConfig({
            "experiment_name": "test_save_load",
            "num_rounds": 7,
            "learning_rate": 0.005
        })

        filepath = manager.save_config(
            config, "test_config.json"
        )
        assert os.path.exists(filepath)

        loaded = manager.load_config(filepath)
        assert loaded.get("num_rounds") == 7
        assert loaded.get("learning_rate") == 0.005

        print("✅ PASS: Config saved and loaded correctly")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    finally:
        shutil.rmtree(TEST_CONFIG_DIR, ignore_errors=True)


def test_file_not_found_error():
    """Test loading non-existent config raises error"""
    print("\nTest 10: File not found error")
    manager = ConfigManager(TEST_CONFIG_DIR)
    try:
        manager.load_config(
            "data/configs/nonexistent.json"
        )
        print("❌ FAIL: Should have raised error")
    except FileNotFoundError as e:
        print(f"✅ PASS: FileNotFoundError raised")
    finally:
        shutil.rmtree(TEST_CONFIG_DIR, ignore_errors=True)


def test_config_to_dict():
    """Test config serialisation to dict"""
    print("\nTest 11: Config to dict serialisation")
    config = ExperimentConfig()
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert "num_rounds" in config_dict
    assert "learning_rate" in config_dict
    assert "languages" in config_dict

    print("✅ PASS: Config serialisation correct")


def test_config_summary_export():
    """Test config summary export"""
    print("\nTest 12: Config summary export")
    os.makedirs(TEST_CONFIG_DIR, exist_ok=True)

    try:
        manager = ConfigManager(TEST_CONFIG_DIR)
        config = ExperimentConfig()
        summary = manager.export_config_summary(config)

        assert isinstance(summary, str)
        assert "num_rounds" in summary
        assert "learning_rate" in summary

        print("✅ PASS: Config summary exported correctly")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    finally:
        shutil.rmtree(TEST_CONFIG_DIR, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 55)
    print("Running SBFLT-24 Experiment Config Tests")
    print("=" * 55)

    test_default_config_valid()
    test_custom_config_valid()
    test_zero_rounds_rejected()
    test_negative_learning_rate_rejected()
    test_invalid_partition_type_rejected()
    test_invalid_language_rejected()
    test_min_clients_exceeds_num_clients()
    test_learning_rate_exceeds_one()
    test_save_and_load_config()
    test_file_not_found_error()
    test_config_to_dict()
    test_config_summary_export()

    print("\n" + "=" * 55)
    print("All SBFLT-24 tests completed")
    print("=" * 55)