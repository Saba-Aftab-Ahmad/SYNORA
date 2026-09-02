"""
SBFLT-9: Dataset Loader Module
Loads low-resource Kenyan language datasets from CSV files.
Handles Dholuo, Kalenjin, and Kidawida language pairs.

Author: Team Member 1
Sprint: 1
"""

import csv
import os


# Supported languages in the system
SUPPORTED_LANGUAGES = ["dholuo", "kalenjin", "kidawida"]

# Base path for datasets
DATASET_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "datasets"
)


def load_dataset(language):
    """
    Load a single language dataset from CSV file.

    Args:
        language (str): Language name. Must be one of
                       'dholuo', 'kalenjin', 'kidawida'

    Returns:
        list: List of (source_text, label) tuples where
              label is the source language name

    Raises:
        ValueError: If language is not supported
        FileNotFoundError: If CSV file does not exist
    """
    # Validate language input
    language = language.lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported: {SUPPORTED_LANGUAGES}"
        )

    # Build file path
    if language == "dholuo":
        filename = "dholuo_swahili.csv"
    elif language == "kalenjin":
        filename = "kalenjin_swahili.csv"
    elif language == "kidawida":
        filename = "kidawida_swahili.csv"

    filepath = os.path.join(
        DATASET_BASE_PATH, language, filename
    )

    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}"
        )

    # Load and parse CSV
    dataset = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_text = row.get(
                "source_text", ""
            ).strip()
            label = row.get(
                "source_language", language
            ).strip()

            # Skip empty rows
            if source_text:
                dataset.append((source_text, label))

    print(
        f"Loaded {len(dataset)} samples "
        f"from {language} dataset"
    )
    return dataset


def load_all_datasets():
    """
    Load all three language datasets and combine them.

    Returns:
        list: Combined list of (source_text, label) tuples
              from all three languages
    """
    all_data = []

    for language in SUPPORTED_LANGUAGES:
        try:
            data = load_dataset(language)
            all_data.extend(data)
            print(
                f"Successfully loaded: {language} "
                f"({len(data)} samples)"
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        except Exception as e:
            print(f"Error loading {language}: {e}")

    print(
        f"\nTotal samples loaded: {len(all_data)}"
    )
    return all_data


def get_dataset_info(dataset):
    """
    Returns summary information about a loaded dataset.

    Args:
        dataset: List of (text, label) tuples

    Returns:
        dict: Summary with total count, label counts,
              and sample texts
    """
    if not dataset:
        return {"error": "Dataset is empty"}

    # Count samples per label
    label_counts = {}
    for text, label in dataset:
        label_counts[label] = (
            label_counts.get(label, 0) + 1
        )

    info = {
        "total_samples": len(dataset),
        "num_classes": len(label_counts),
        "label_counts": label_counts,
        "sample_texts": [
            dataset[0][0],
            dataset[len(dataset)//2][0],
            dataset[-1][0]
        ]
    }

    return info


def validate_dataset(dataset):
    """
    Validates a loaded dataset for quality checks.

    Args:
        dataset: List of (text, label) tuples

    Returns:
        dict: Validation report
    """
    if not dataset:
        return {
            "valid": False,
            "error": "Dataset is empty"
        }

    total = len(dataset)
    empty_texts = sum(
        1 for text, label in dataset
        if not text.strip()
    )
    missing_labels = sum(
        1 for text, label in dataset
        if not label.strip()
    )

    report = {
        "valid": empty_texts == 0 and missing_labels == 0,
        "total_samples": total,
        "empty_texts": empty_texts,
        "missing_labels": missing_labels,
        "unique_labels": list(set(
            label for text, label in dataset
        ))
    }

    return report