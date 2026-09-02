"""
Unit Tests for SBFLT-9: Dataset Loader
Tests loading, validation, and info functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from preprocessing.loader import (
    load_dataset,
    load_all_datasets,
    get_dataset_info,
    validate_dataset
)


def test_load_single_dataset():
    """Test loading a single language dataset"""
    print("\nTest: Load single dataset (dholuo)")
    dataset = load_dataset("dholuo")

    assert len(dataset) > 0, "Dataset should not be empty"
    assert isinstance(dataset[0], tuple), \
        "Each item should be a tuple"
    assert len(dataset[0]) == 2, \
        "Each tuple should have 2 elements (text, label)"

    print(f"PASS: Loaded {len(dataset)} samples")


def test_load_all_datasets():
    """Test loading all three language datasets"""
    print("\nTest: Load all datasets")
    all_data = load_all_datasets()

    assert len(all_data) > 0, \
        "Combined dataset should not be empty"

    labels = set(label for text, label in all_data)
    print(f"PASS: Total samples: {len(all_data)}")
    print(f"PASS: Languages found: {labels}")


def test_dataset_info():
    """Test dataset info summary"""
    print("\nTest: Dataset info")
    dataset = load_dataset("kalenjin")
    info = get_dataset_info(dataset)

    assert "total_samples" in info
    assert "label_counts" in info
    assert info["total_samples"] > 0

    print(f"PASS: Info generated correctly")
    print(f"      Total: {info['total_samples']}")
    print(f"      Labels: {info['label_counts']}")


def test_validate_dataset():
    """Test dataset validation"""
    print("\nTest: Dataset validation")
    dataset = load_dataset("kidawida")
    report = validate_dataset(dataset)

    assert "valid" in report
    assert report["total_samples"] > 0

    print(f"PASS: Validation report generated")
    print(f"      Valid: {report['valid']}")
    print(f"      Empty texts: {report['empty_texts']}")


def test_invalid_language():
    """Test that invalid language raises error"""
    print("\nTest: Invalid language handling")
    try:
        load_dataset("french")
        print("FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"PASS: Correctly raised ValueError: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Running SBFLT-9 Loader Tests")
    print("=" * 50)

    test_load_single_dataset()
    test_load_all_datasets()
    test_dataset_info()
    test_validate_dataset()
    test_invalid_language()

    print("\n" + "=" * 50)
    print("All SBFLT-9 tests completed")
    print("=" * 50)