"""
US-03: Automated Text Preprocessing Pipeline
Synora - Browser-Based Federated Learning Toolkit

This module handles all text preprocessing:
- Tokenization
- Vocabulary building
- Sequence encoding
- Padding to uniform length
- OOV (out-of-vocabulary) handling
- Output validation (no nulls, consistent shape)

Run ONCE on the server to build vocabulary and preprocessed shards.
Each client shard is saved as a JSON file ready to load in the browser.
"""

import json
import re
import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these as needed)
# ─────────────────────────────────────────────

MAX_SEQUENCE_LENGTH = 50      # All sequences padded / truncated to this length
MIN_WORD_FREQUENCY  = 1       # Words appearing fewer times are treated as OOV
PADDING_TOKEN       = "<PAD>" # Index 0 — used for padding
OOV_TOKEN           = "<OOV>" # Index 1 — used for unknown words

# Paths
DATA_DIR   = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "preprocessed"
VOCAB_PATH = OUTPUT_DIR / "vocabulary.json"


# ─────────────────────────────────────────────
#  STEP 1 — TOKENIZER
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Tokenize a single text string into a list of lowercase word tokens.

    What it does:
      - Lowercases everything
      - Removes punctuation (keeps letters, digits, spaces)
      - Splits on whitespace
      - Filters out empty strings

    Args:
        text: Raw text string, e.g. "Habari yako leo!"

    Returns:
        List of token strings, e.g. ["habari", "yako", "leo"]
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Lowercase
    text = text.lower().strip()

    # Remove punctuation but keep letters (including unicode for Swahili, Luo etc.)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Split and filter
    tokens = [t for t in text.split() if t]

    return tokens


def tokenize_dataset(samples: list[dict]) -> list[dict]:
    """
    Apply tokenize() to every sample in a dataset.

    Args:
        samples: List of dicts with at least a "text" key.

    Returns:
        Same list but each dict now has a "tokens" key added.
    """
    tokenized = []
    for i, sample in enumerate(samples):
        tokens = tokenize(sample.get("text", ""))
        tokenized.append({
            **sample,
            "tokens": tokens
        })

    print(f"[Tokenizer] Tokenized {len(tokenized)} samples.")
    return tokenized


# ─────────────────────────────────────────────
#  STEP 2 — VOCABULARY BUILDER
# ─────────────────────────────────────────────

def build_vocabulary(tokenized_samples: list[dict]) -> dict[str, int]:
    """
    Build a vocabulary mapping from word tokens to integer indices.

    Special tokens:
      - <PAD> → 0   (used to pad short sequences)
      - <OOV> → 1   (used for unknown words at inference time)

    All real words start from index 2.

    Args:
        tokenized_samples: Output of tokenize_dataset()

    Returns:
        vocab: dict mapping word → integer index
               e.g. {"<PAD>": 0, "<OOV>": 1, "habari": 2, ...}
    """
    # Count word frequencies
    freq: dict[str, int] = {}
    for sample in tokenized_samples:
        for token in sample.get("tokens", []):
            freq[token] = freq.get(token, 0) + 1

    # Build vocab: only include words that meet minimum frequency
    vocab = {PADDING_TOKEN: 0, OOV_TOKEN: 1}
    index = 2
    for word, count in sorted(freq.items()):  # sorted for reproducibility
        if count >= MIN_WORD_FREQUENCY:
            vocab[word] = index
            index += 1

    print(f"[Vocabulary] Built vocabulary with {len(vocab)} tokens "
          f"(including {PADDING_TOKEN} and {OOV_TOKEN}).")
    return vocab


def save_vocabulary(vocab: dict[str, int], path: Path = VOCAB_PATH) -> None:
    """Save vocabulary dict to a JSON file so clients can load it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[Vocabulary] Saved to {path}")


def load_vocabulary(path: Path = VOCAB_PATH) -> dict[str, int]:
    """Load a previously saved vocabulary from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[Vocabulary] Loaded {len(vocab)} tokens from {path}")
    return vocab


# ─────────────────────────────────────────────
#  STEP 3 — SEQUENCE ENCODER  (tokens → ints)
# ─────────────────────────────────────────────

def encode_sequence(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """
    Convert a list of string tokens to a list of integer indices
    using the vocabulary. Unknown tokens map to OOV index (1).

    Args:
        tokens: e.g. ["habari", "yako", "unknownword"]
        vocab:  vocabulary dict

    Returns:
        e.g. [2, 3, 1]   ← unknownword → OOV index 1
    """
    oov_index = vocab[OOV_TOKEN]
    return [vocab.get(token, oov_index) for token in tokens]


def encode_dataset(tokenized_samples: list[dict],
                   vocab: dict[str, int]) -> list[dict]:
    """
    Apply encode_sequence() to every sample.

    Args:
        tokenized_samples: Output of tokenize_dataset()
        vocab: vocabulary dict

    Returns:
        Same list with a "sequence" key added (list of ints).
    """
    encoded = []
    for sample in tokenized_samples:
        sequence = encode_sequence(sample.get("tokens", []), vocab)
        encoded.append({
            **sample,
            "sequence": sequence
        })
    print(f"[Encoder] Encoded {len(encoded)} samples to integer sequences.")
    return encoded


# ─────────────────────────────────────────────
#  STEP 4 — PADDING
# ─────────────────────────────────────────────

def pad_sequence(sequence: list[int],
                 max_length: int = MAX_SEQUENCE_LENGTH,
                 pad_index: int = 0) -> list[int]:
    """
    Pad or truncate a sequence to exactly max_length integers.

    - If shorter → append pad_index (0) on the right
    - If longer  → truncate to first max_length tokens

    Args:
        sequence:   List of ints (encoded tokens)
        max_length: Target length for all sequences
        pad_index:  Integer to use for padding (default 0 = <PAD>)

    Returns:
        A list of exactly max_length integers.
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        padding = [pad_index] * (max_length - len(sequence))
        return sequence + padding


def pad_dataset(encoded_samples: list[dict],
                max_length: int = MAX_SEQUENCE_LENGTH) -> list[dict]:
    """
    Apply pad_sequence() to every sample.

    Args:
        encoded_samples: Output of encode_dataset()
        max_length: Uniform target length

    Returns:
        Same list with "padded_sequence" key added.
    """
    padded = []
    for sample in encoded_samples:
        padded_seq = pad_sequence(sample.get("sequence", []), max_length)
        padded.append({
            **sample,
            "padded_sequence": padded_seq
        })
    print(f"[Padding] Padded all sequences to length {max_length}.")
    return padded


# ─────────────────────────────────────────────
#  STEP 5 — LABEL ENCODER
# ─────────────────────────────────────────────

def build_label_map(samples: list[dict]) -> dict[str, int]:
    """
    Build a mapping from string labels to integer class indices.

    Args:
        samples: List of dicts with a "label" key.

    Returns:
        e.g. {"greeting": 0, "thanks": 1, "complaint": 2}
    """
    unique_labels = sorted(set(s["label"] for s in samples if "label" in s))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"[Labels] Found {len(label_map)} classes: {label_map}")
    return label_map


def encode_labels(samples: list[dict],
                  label_map: dict[str, int]) -> list[dict]:
    """
    Add an "encoded_label" integer field to each sample.
    """
    encoded = []
    for sample in samples:
        label_str = sample.get("label", "")
        encoded_label = label_map.get(label_str, -1)
        encoded.append({**sample, "encoded_label": encoded_label})
    return encoded


# ─────────────────────────────────────────────
#  STEP 6 — VALIDATION
# ─────────────────────────────────────────────

def validate_output(padded_samples: list[dict],
                    max_length: int = MAX_SEQUENCE_LENGTH) -> bool:
    """
    Validate the final preprocessed output against all US-03 acceptance criteria:

    ✅ All sequences have uniform length (= max_length)
    ✅ No null / None values in any sequence
    ✅ No NaN values
    ✅ All encoded_labels are valid non-negative integers
    ✅ Dataset is non-empty

    Args:
        padded_samples: Final preprocessed samples
        max_length: Expected sequence length

    Returns:
        True if all checks pass. Raises ValueError on first failure.
    """
    print("\n[Validation] Running acceptance criteria checks...")

    if not padded_samples:
        raise ValueError("❌ FAIL: Output dataset is empty.")

    for i, sample in enumerate(padded_samples):
        seq = sample.get("padded_sequence")

        # Check sequence exists
        if seq is None:
            raise ValueError(f"❌ FAIL: Sample {i} has null padded_sequence.")

        # Check uniform length
        if len(seq) != max_length:
            raise ValueError(
                f"❌ FAIL: Sample {i} has length {len(seq)}, expected {max_length}."
            )

        # Check no None values inside sequence
        if any(v is None for v in seq):
            raise ValueError(f"❌ FAIL: Sample {i} contains None inside sequence.")

        # Check no NaN (can happen with float conversion)
        if any(isinstance(v, float) and np.isnan(v) for v in seq):
            raise ValueError(f"❌ FAIL: Sample {i} contains NaN inside sequence.")

        # Check all values are non-negative integers
        if any(not isinstance(v, int) or v < 0 for v in seq):
            raise ValueError(
                f"❌ FAIL: Sample {i} contains non-integer or negative value."
            )

        # Check encoded label
        label = sample.get("encoded_label", -1)
        if label < 0:
            raise ValueError(f"❌ FAIL: Sample {i} has invalid encoded_label {label}.")

    print(f"✅ PASS: All {len(padded_samples)} samples validated.")
    print(f"✅ PASS: Uniform sequence length = {max_length}")
    print(f"✅ PASS: No null values in output tensors")
    print(f"✅ PASS: OOV tokens handled (mapped to index 1)")
    print(f"✅ PASS: All labels are valid integers")
    return True


# ─────────────────────────────────────────────
#  STEP 7 — SAVE PREPROCESSED SHARDS
# ─────────────────────────────────────────────

def save_shard(shard: list[dict], client_id: int,
               label_map: dict[str, int],
               output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Save a single client's preprocessed shard to JSON.
    The JSON contains only the data the client needs — no raw text is exposed.

    Structure saved:
      {
        "client_id": 0,
        "num_samples": 120,
        "sequence_length": 50,
        "label_map": {"greeting": 0, ...},
        "samples": [
          {"padded_sequence": [2, 3, 0, ...], "encoded_label": 1},
          ...
        ]
      }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"client_{client_id}_preprocessed.json"

    payload = {
        "client_id": client_id,
        "num_samples": len(shard),
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "label_map": label_map,
        "samples": [
            {
                "padded_sequence": s["padded_sequence"],
                "encoded_label":   s["encoded_label"]
            }
            for s in shard
        ]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"[Save] Client {client_id} shard → {path} ({len(shard)} samples)")
    return path


# ─────────────────────────────────────────────
#  MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(
    client_shards: dict[int, list[dict]]
) -> dict:
    """
    Full US-03 preprocessing pipeline.

    Takes raw client shards (output of US-02 partitioning),
    applies the complete preprocessing pipeline, validates the
    output, and saves ready-to-use JSON files per client.

    Args:
        client_shards: dict mapping client_id → list of raw samples
                       Each sample must have "text" and "label" keys.
                       e.g. {0: [{"text": "Habari", "label": "greeting"}, ...], 1: [...]}

    Returns:
        result dict with vocabulary, label_map, and output paths.
    """
    print("=" * 60)
    print("  US-03: Automated Text Preprocessing Pipeline — START")
    print("=" * 60)

    # ── Combine all shards to build a global vocabulary ──────────
    # IMPORTANT: vocab must be built from ALL data so every client
    # uses the same word→index mapping (acceptance criteria: consistent)
    all_samples = []
    for shard in client_shards.values():
        all_samples.extend(shard)

    print(f"\n[Pipeline] Total samples across all shards: {len(all_samples)}")

    # Step 1: Tokenize all samples
    all_tokenized = tokenize_dataset(all_samples)

    # Step 2: Build shared vocabulary
    vocab = build_vocabulary(all_tokenized)
    save_vocabulary(vocab)

    # Step 3: Build label map
    label_map = build_label_map(all_samples)

    # Save label map alongside vocabulary
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Labels] Saved label map to {label_map_path}")

    # ── Process each client shard independently ───────────────────
    output_paths = {}

    for client_id, raw_shard in client_shards.items():
        print(f"\n── Processing Client {client_id} ({len(raw_shard)} samples) ──")

        # Step 1: Tokenize this shard
        tokenized = tokenize_dataset(raw_shard)

        # Step 2: Encode tokens → integers using SHARED vocabulary
        encoded = encode_dataset(tokenized, vocab)

        # Step 3: Encode labels → integers
        labeled = encode_labels(encoded, label_map)

        # Step 4: Pad to uniform length
        padded = pad_dataset(labeled, MAX_SEQUENCE_LENGTH)

        # Step 5: Validate output (acceptance criteria checks)
        validate_output(padded, MAX_SEQUENCE_LENGTH)

        # Step 6: Save preprocessed shard
        path = save_shard(padded, client_id, label_map)
        output_paths[client_id] = str(path)

    print("\n" + "=" * 60)
    print("  US-03: Preprocessing Pipeline — COMPLETE ✅")
    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Sequence length : {MAX_SEQUENCE_LENGTH}")
    print(f"  Clients processed: {list(output_paths.keys())}")
    print("=" * 60)

    return {
        "vocabulary": vocab,
        "label_map": label_map,
        "vocab_size": len(vocab),
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "output_paths": output_paths
    }


# ─────────────────────────────────────────────
#  QUICK-TEST  (run this file directly to test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal sample data simulating what US-02 partitioning produces
    mock_shards = {
        0: [
            {"text": "Habari yako leo asubuhi", "label": "greeting"},
            {"text": "Habari za asubuhi rafiki", "label": "greeting"},
            {"text": "Nzuri sana asante wewe", "label": "positive"},
            {"text": "Mimi ninasikia njaa sana leo", "label": "complaint"},
            {"text": "Tafadhali nisaidie na hili tatizo", "label": "request"},
        ],
        1: [
            {"text": "Asante sana kwa msaada wako", "label": "positive"},
            {"text": "Sijui la kufanya sasa hivi", "label": "complaint"},
            {"text": "Ninahitaji chakula na maji sasa", "label": "request"},
            {"text": "Habari za jioni rafiki yangu", "label": "greeting"},
            {"text": "Mambo mazuri sana leo hii", "label": "positive"},
        ],
        2: [
            {"text": "Tafadhali niambie ukweli wote", "label": "request"},
            {"text": "Ninafurahi sana kukuona tena", "label": "positive"},
            {"text": "Kuna tatizo kubwa sana hapa", "label": "complaint"},
            {"text": "Habari nzuri rafiki zangu wote", "label": "greeting"},
            {"text": "Asante kwa kila kitu ulichofanya", "label": "positive"},
        ],
    }

    result = run_preprocessing_pipeline(mock_shards)
    print("\nPipeline result summary:")
    print(f"  Vocab size     : {result['vocab_size']}")
    print(f"  Sequence length: {result['sequence_length']}")
    print(f"  Label map      : {result['label_map']}")
    print(f"  Output files   : {list(result['output_paths'].values())}")
