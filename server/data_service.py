"""
server/data_service.py
======================
Dataset Partitioning Module + Text Preprocessing Pipeline, wired into the
live coordination server (SDS 3.1.1.4, 3.1.1.6, 3.2.1 steps 2-3).

What this module does
---------------------
1. Loads the three Kenyan-language CSV files (Dholuo / Kalenjin / Kidawida)
   through preprocessing.loader.load_dataset  (US-01).
2. Draws, with a FIXED seed, exactly SAMPLES_PER_CLIENT (=2000) training
   sentences per language and a small held-out validation sample per
   language that is NEVER used for training (UC-10b: server-held validation
   set).
3. Builds ONE shared vocabulary (top-N most frequent tokens, <PAD>=0,
   <OOV>=1) with preprocessing.preprocessor and converts every sentence
   into a fixed-length integer sequence (US-03).
4. Produces the per-client shard according to the session's partition type
   (US-02).  A client can join at ANY time, so a shard is computed on demand
   from the client's join index alone (no "number of clients" is needed):
        "iid"       - 2000 uniformly random sentences of the mixed pool; clients
                      0,1,2 get disjoint thirds, clients 3,4,5 another disjoint
                      split with a different shuffle, and so on.
        "language"  - client i owns language (i mod 3)  (natural non-IID).
        "dirichlet" - class mix drawn from Dirichlet(alpha), 2000 sentences.

TASK DEFINITION (important for the report / viva):
    The task is TEXT CLASSIFICATION - given a sentence written in an
    unknown Kenyan language, predict WHICH language it is
    (Dholuo / Kalenjin / Kidawida).  The Swahili column (`target_text`) of
    the CSV files is NOT used: NO machine translation takes place anywhere
    in Synora.

Only integer sequences and integer labels ever leave this module; the
browser never needs the raw sentences.
"""

import threading
from typing import Dict, List, Tuple

import numpy as np

from preprocessing.loader import load_dataset
from preprocessing.preprocessor import (
    OOV_TOKEN,
    PADDING_TOKEN,
    build_vocabulary,
    pad_sequence,
    tokenize,
    tokenize_dataset,
)
from server import model_spec

# language key (CSV folder name) -> label index (== position in model_spec.LABELS)
LANGUAGE_KEYS = ["dholuo", "kalenjin", "kidawida"]

# Legacy partition names used by the first version of the frontend/backend.
PARTITION_TO_LANGUAGE = {
    "luo_swa": "Dholuo",
    "kln_swa": "Kalenjin",
    "dav_swa": "Kidawida",
}
LANGUAGE_TO_PARTITION = {v: k for k, v in PARTITION_TO_LANGUAGE.items()}

VALIDATION_PER_LANGUAGE = 300
DATA_SEED = 42

_lock = threading.Lock()
_cache: Dict[str, object] = {}


# ── Preprocessing helpers ────────────────────────────────────────────────

def encode_text(text: str, vocab: Dict[str, int], max_length: int = model_spec.SEQUENCE_LENGTH) -> List[int]:
    """tokenise -> map to vocabulary ids (OOV=1) -> pad/truncate (post-padding)."""
    oov = vocab[OOV_TOKEN]
    ids = [vocab.get(tok, oov) for tok in tokenize(text)]
    return pad_sequence(ids, max_length, vocab[PADDING_TOKEN])


def _load_all() -> dict:
    rng = np.random.default_rng(DATA_SEED)
    train_texts: List[str] = []
    train_labels: List[int] = []
    val_texts: List[str] = []
    val_labels: List[int] = []

    for label_idx, lang_key in enumerate(LANGUAGE_KEYS):
        rows = [r for r in load_dataset(lang_key) if r[0] and r[0].strip()]
        order = rng.permutation(len(rows))
        n_train = min(model_spec.SAMPLES_PER_CLIENT, len(rows))
        train_sel = order[:n_train]
        val_sel = order[n_train : n_train + VALIDATION_PER_LANGUAGE]
        for i in train_sel:
            train_texts.append(rows[int(i)][0])
            train_labels.append(label_idx)
        for i in val_sel:
            val_texts.append(rows[int(i)][0])
            val_labels.append(label_idx)

    # Shared vocabulary, built from TRAINING sentences only (no leakage).
    tokenized = tokenize_dataset([{"text": t} for t in train_texts])
    vocab = build_vocabulary(tokenized, max_vocab_size=model_spec.VOCAB_SIZE)

    def encode_many(texts):
        return np.asarray([encode_text(t, vocab) for t in texts], dtype=np.int32)

    train_x = encode_many(train_texts)
    val_x = encode_many(val_texts)
    train_y = np.asarray(train_labels, dtype=np.int32)
    val_y = np.asarray(val_labels, dtype=np.int32)

    validate_preprocessed(train_x, train_y)
    validate_preprocessed(val_x, val_y)

    oov_rate = float(np.mean(train_x == vocab[OOV_TOKEN]))
    print(
        f"[DataService] train={len(train_y)} val={len(val_y)} vocab={len(vocab)} "
        f"seq_len={model_spec.SEQUENCE_LENGTH} oov_rate={oov_rate:.3f}"
    )
    return {
        "vocab": vocab,
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
    }


def validate_preprocessed(x: np.ndarray, y: np.ndarray) -> None:
    """SDS validatePreprocessedOutput: shape, no nulls / NaN, valid labels."""
    if x.size == 0:
        raise ValueError("Preprocessing produced an empty dataset")
    if x.ndim != 2 or x.shape[1] != model_spec.SEQUENCE_LENGTH:
        raise ValueError(f"Bad sequence shape {x.shape}")
    if x.min() < 0 or x.max() >= model_spec.VOCAB_SIZE:
        raise ValueError("Token id outside vocabulary range")
    if y.min() < 0 or y.max() >= model_spec.NUM_CLASSES:
        raise ValueError("Label outside class range")


def get_data() -> dict:
    """Lazily build (once per worker process) and return the cached data."""
    if not _cache:
        with _lock:
            if not _cache:
                _cache.update(_load_all())
    return _cache


def reset_cache() -> None:  # used by tests
    with _lock:
        _cache.clear()


# ── Validation set (server-held) ─────────────────────────────────────────

def validation_set() -> Tuple[np.ndarray, np.ndarray]:
    d = get_data()
    return d["val_x"], d["val_y"]


# ── Per-client shards ────────────────────────────────────────────────────

def _iid_indices(partition_index: int, seed: int) -> np.ndarray:
    """Disjoint random thirds of the pool; every further block of 3 clients is re-shuffled."""
    pool = len(get_data()["train_y"])
    size = min(model_spec.SAMPLES_PER_CLIENT, pool)
    slots = max(1, pool // size)
    block, slot = divmod(int(partition_index), slots)
    order = np.random.default_rng([int(seed), block]).permutation(pool)
    return np.sort(order[slot * size:(slot + 1) * size])


def _dirichlet_indices(partition_index: int, alpha: float, seed: int) -> np.ndarray:
    """SAMPLES_PER_CLIENT sentences whose language mix is drawn from Dirichlet(alpha)."""
    train_y = get_data()["train_y"]
    size = min(model_spec.SAMPLES_PER_CLIENT, len(train_y))
    rng = np.random.default_rng([int(seed), 7919, int(partition_index)])
    by_class = [np.where(train_y == k)[0] for k in range(model_spec.NUM_CLASSES)]
    for _ in range(50):
        p = rng.dirichlet([float(alpha)] * model_spec.NUM_CLASSES)
        counts = np.floor(p * size).astype(int)
        counts[int(np.argmax(p))] += size - counts.sum()      # make the total exactly `size`
        if all(counts[k] <= len(by_class[k]) for k in range(model_spec.NUM_CLASSES)):
            break
    else:                                                      # extremely skewed draw: clip to what exists
        counts = np.minimum(counts, [len(b) for b in by_class])
    parts = [rng.choice(by_class[k], size=int(counts[k]), replace=False) for k in range(model_spec.NUM_CLASSES)]
    return np.sort(np.concatenate(parts))


def client_shard(
    partition_type: str,
    partition_index: int,
    language: str,
    alpha: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (sequences[n, L], labels[n]) for one client, never more than
    SAMPLES_PER_CLIENT samples.  Deterministic: the same client always gets the
    same shard, whatever order or time the other clients join.
    """
    d = get_data()
    train_x, train_y = d["train_x"], d["train_y"]

    if partition_type == "language":
        idx = np.where(train_y == model_spec.LABELS.index(language))[0]
    elif partition_type == "dirichlet":
        idx = _dirichlet_indices(partition_index, alpha, seed)
    else:
        idx = _iid_indices(partition_index, seed)

    if len(idx) > model_spec.SAMPLES_PER_CLIENT:
        rng = np.random.default_rng(seed + 1000 + partition_index)
        idx = np.sort(rng.choice(idx, size=model_spec.SAMPLES_PER_CLIENT, replace=False))
    return train_x[idx], train_y[idx]


def shard_label_distribution(y: np.ndarray) -> Dict[str, int]:
    """Label histogram for logging / dashboard (SDS validatePartitions)."""
    return {model_spec.LABELS[k]: int(np.sum(y == k)) for k in range(model_spec.NUM_CLASSES)}


def language_for_partition(partition: str) -> str:
    return PARTITION_TO_LANGUAGE.get(partition, "")
