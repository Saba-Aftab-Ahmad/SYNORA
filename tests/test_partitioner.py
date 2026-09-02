"""
Unit Tests for SBFLT-10: Dataset Partitioner
Tests IID, Non-IID, validation, and distribution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from preprocessing.loader import load_all_datasets
from preprocessing.partitioner import (
    partition_iid,
    partition_non_iid,
    validate_partitions,
    get_label_distribution,
    save_partitions
)


def get_test_dataset():
    """Load real dataset for testing"""
    return load_all_datasets()


def test_iid_partition_count():
    """Test IID creates correct number of partitions"""
    print("\nTest: IID partition count")
    dataset = get_test_dataset()
    partitions = partition_iid(dataset, num_clients=5)

    assert len(partitions) == 5, \
        "Should create exactly 5 partitions"
    print("PASS: Correct number of partitions created")


def test_iid_total_size():
    """Test IID assigns all samples"""
    print("\nTest: IID total size integrity")
    dataset = get_test_dataset()
    partitions = partition_iid(dataset, num_clients=5)
    report = validate_partitions(partitions, dataset)

    assert report["sizes_match"] == True, \
        "Total assigned should match original size"
    print("PASS: All samples correctly assigned in IID")


def test_iid_equal_sizes():
    """Test IID partitions have approximately equal sizes"""
    print("\nTest: IID equal partition sizes")
    dataset = get_test_dataset()
    partitions = partition_iid(dataset, num_clients=5)

    sizes = [len(p) for p in partitions.values()]
    size_diff = max(sizes) - min(sizes)

    assert size_diff <= 5, \
        "IID partitions should have similar sizes"
    print(
        f"PASS: Size difference between clients: {size_diff}"
    )


def test_non_iid_total_size():
    """Test Non-IID assigns all samples"""
    print("\nTest: Non-IID total size integrity")
    dataset = get_test_dataset()
    partitions = partition_non_iid(
        dataset, num_clients=5, alpha=0.5
    )
    report = validate_partitions(partitions, dataset)

    assert report["sizes_match"] == True, \
        "Total assigned should match original size"
    print("PASS: All samples correctly assigned in Non-IID")


def test_non_iid_skew():
    """Test Non-IID produces label skew"""
    print("\nTest: Non-IID label skew")
    dataset = get_test_dataset()
    partitions = partition_non_iid(
        dataset, num_clients=5, alpha=0.1
    )
    distribution = get_label_distribution(partitions)

    skewed = False
    for client_id, label_counts in distribution.items():
        counts = list(label_counts.values())
        if len(counts) > 0:
            dominant = max(counts) / sum(counts)
            if dominant > 0.6:
                skewed = True
                break

    assert skewed == True, \
        "Non-IID with alpha=0.1 should produce skew"
    print("PASS: Non-IID correctly produces label skew")


def test_no_data_leakage():
    """Test no overlap between client partitions"""
    print("\nTest: No data leakage between partitions")
    dataset = get_test_dataset()
    partitions = partition_iid(dataset, num_clients=5)
    report = validate_partitions(partitions, dataset)

    assert report["no_duplicates"] == True, \
        "No sample should appear in multiple partitions"
    print("PASS: No data leakage detected")


def test_save_partitions():
    """Test saving partitions to files"""
    print("\nTest: Save partitions to files")
    dataset = get_test_dataset()
    partitions = partition_iid(dataset, num_clients=3)

    save_partitions(
        partitions,
        output_dir="data/partitions"
    )

    for i in range(3):
        filepath = f"data/partitions/client_{i}_partition.txt"
        assert os.path.exists(filepath), \
            f"Partition file {filepath} should exist"

    print("PASS: Partition files saved successfully")


if __name__ == "__main__":
    print("=" * 50)
    print("Running SBFLT-10 Partitioner Tests")
    print("=" * 50)

    test_iid_partition_count()
    test_iid_total_size()
    test_iid_equal_sizes()
    test_non_iid_total_size()
    test_non_iid_skew()
    test_no_data_leakage()
    test_save_partitions()

    print("\n" + "=" * 50)
    print("All SBFLT-10 tests completed")
    print("=" * 50)