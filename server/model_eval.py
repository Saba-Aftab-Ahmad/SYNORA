"""
server/model_eval.py
====================
Real evaluation of the aggregated global model on the SERVER-HELD validation
set (SDS 3.1.2.3 evaluateGlobalModel, SRS UC-10b, SDS Table 11
EVALUATION_RESULT: accuracy, precision, recall, f1_score, loss).

The model is tiny, so the forward pass is done in plain numpy and is an exact
mirror of the TF.js model:

    ids[N, L] -> Embedding -> mean over L (GlobalAveragePooling1D)
              -> Dense(relu) -> Dense(softmax)        (dropout = off)

This replaces the old `validate_global_model` that "simulated" predictions
from weight magnitudes.
"""

from typing import Dict, List

import numpy as np

from server import model_spec


def forward(weights: List[np.ndarray], x_ids: np.ndarray) -> np.ndarray:
    """Return class probabilities, shape [N, NUM_CLASSES]."""
    emb, w1, b1, w2, b2 = [np.asarray(w, dtype=np.float32) for w in weights]
    h = emb[x_ids].mean(axis=1)                     # [N, E]  (includes <PAD> positions, like TF.js)
    h = np.maximum(h @ w1 + b1, 0.0)                # relu
    logits = h @ w2 + b2
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(weights: List[np.ndarray], x_ids: np.ndarray, y_true: np.ndarray) -> Dict:
    """Accuracy, cross-entropy loss, macro precision / recall / F1, per-class stats."""
    probs = forward(weights, x_ids)
    n = len(y_true)
    pred = probs.argmax(axis=1)
    eps = 1e-7
    loss = float(-np.mean(np.log(probs[np.arange(n), y_true] + eps)))
    accuracy = float(np.mean(pred == y_true))

    per_class = []
    precisions, recalls, f1s = [], [], []
    for k, name in enumerate(model_spec.LABELS):
        tp = int(np.sum((pred == k) & (y_true == k)))
        fp = int(np.sum((pred == k) & (y_true != k)))
        fn = int(np.sum((pred != k) & (y_true == k)))
        support = int(np.sum(y_true == k))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        per_class.append(
            {
                "language": name,
                "accuracy": r,            # per-class accuracy == recall
                "precision": p,
                "recall": r,
                "f1": f1,
                "support": support,
            }
        )

    return {
        "accuracy": accuracy,
        "loss": loss,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "per_class": per_class,
        "num_samples": int(n),
    }
