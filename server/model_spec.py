"""
server/model_spec.py
====================
SINGLE SOURCE OF TRUTH for the lightweight text-classification model.

SRS NFR-A03  : serialised weights must not exceed 5 MB.
SDS 10.4     : "a small embedding layer and dense classification layers".
SDS 3.2.1/5  : the coordination server initialises the global model weights
               and distributes the SAME initial weights to every client.

The browser client (lib/fl_model.ts) builds its TF.js model from the
`architecture` dictionary returned by the server, so nothing about the model
is hard-coded twice.

Architecture (identical on client and server):

    Embedding(5000 x 32)  ->  GlobalAveragePooling1D  ->  Dense(16, relu)
        ->  Dropout(0.3)  ->  Dense(3, softmax)

    Parameters: 160,000 + 528 + 51 = 160,579  (~0.64 MB as float32)
"""

import numpy as np

# ── Model hyper-structure ────────────────────────────────────────────────
VOCAB_SIZE = 5000        # incl. <PAD>=0 and <OOV>=1
EMBEDDING_DIM = 32
SEQUENCE_LENGTH = 40     # words per sample (99th percentile of the corpus is 40)
HIDDEN_UNITS = 16
DROPOUT_RATE = 0.3
NUM_CLASSES = 3

# Class order MUST be identical for every client (FedAvg averages class k of
# one client with class k of every other client).
LABELS = ["Dholuo", "Kalenjin", "Kidawida"]

# Each simulated client trains on at most this many samples (project rule).
SAMPLES_PER_CLIENT = 2000

# Layer order == order of model.getWeights() in TF.js
LAYER_SPECS = [
    {"layer_name": "embedding/embeddings", "shape": [VOCAB_SIZE, EMBEDDING_DIM]},
    {"layer_name": "dense_1/kernel", "shape": [EMBEDDING_DIM, HIDDEN_UNITS]},
    {"layer_name": "dense_1/bias", "shape": [HIDDEN_UNITS]},
    {"layer_name": "output/kernel", "shape": [HIDDEN_UNITS, NUM_CLASSES]},
    {"layer_name": "output/bias", "shape": [NUM_CLASSES]},
]

ARCHITECTURE_NAME = "embedding_avgpool_dense"

# Hard limit from SRS NFR-A03
MAX_MODEL_BYTES = 5 * 1024 * 1024


def expected_shapes():
    return [tuple(s["shape"]) for s in LAYER_SPECS]


def total_parameters() -> int:
    return int(sum(int(np.prod(s["shape"])) for s in LAYER_SPECS))


def model_size_bytes() -> int:
    """Serialised size of the weights as float32."""
    return total_parameters() * 4


def architecture() -> dict:
    """Architecture description sent to browser clients."""
    return {
        "name": ARCHITECTURE_NAME,
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "max_length": SEQUENCE_LENGTH,
        "hidden_units": HIDDEN_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "num_classes": NUM_CLASSES,
        "labels": LABELS,
        "layers": LAYER_SPECS,
        "total_parameters": total_parameters(),
        "model_size_bytes": model_size_bytes(),
    }


def init_weights(seed: int = 42):
    """
    Create the initial global weights (same initialisers as Keras/TF.js):
      * Embedding        : RandomUniform(-0.05, 0.05)
      * Dense kernels    : GlorotUniform
      * Dense biases     : zeros
    Returns a list of float32 numpy arrays in LAYER_SPECS order.
    """
    rng = np.random.default_rng(int(seed))
    weights = []
    for spec in LAYER_SPECS:
        shape = tuple(spec["shape"])
        name = spec["layer_name"]
        if name.startswith("embedding"):
            w = rng.uniform(-0.05, 0.05, size=shape)
        elif name.endswith("kernel"):
            fan_in, fan_out = shape
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, size=shape)
        else:  # bias
            w = np.zeros(shape)
        weights.append(w.astype(np.float32))
    return weights
