"""
server/weight_codec.py
======================
Weight Serialisation Module (server side) — SDS 3.1.1.5 / 3.1.2.5.

SDS 10.4 "Efficient Weight Serialization": weights travel as float32 typed
arrays instead of nested JSON number lists. We encode each tensor as
base64(float32 little-endian bytes):  a 160k-parameter model is ~0.86 MB
instead of ~3.4 MB of JSON text.

Wire format of ONE tensor:
    {"layer_name": str, "shape": [int, ...], "dtype": "float32", "data_b64": str}

The legacy format  {"shape": [...], "data": [floats]}  is still accepted so
older clients / Postman tests keep working.

This module is also the server half of the Privacy Enforcement Layer
(SDS 3.1.2.7 validateInboundUpdate): it rejects anything that is not a
well-formed numeric tensor with the expected shape and finite values.
"""

import base64
import math
from typing import Any, List, Optional

import numpy as np

from server import model_spec


class PayloadError(ValueError):
    """Raised for structurally invalid / non-compliant weight payloads."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field


# ── Encoding ─────────────────────────────────────────────────────────────

def encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def encode_weights(weights: List[np.ndarray]) -> List[dict]:
    """Numpy weights -> JSON-serialisable list of tensor dicts."""
    out = []
    for spec, w in zip(model_spec.LAYER_SPECS, weights):
        out.append(
            {
                "layer_name": spec["layer_name"],
                "shape": list(w.shape),
                "dtype": "float32",
                "data_b64": encode_array(w),
            }
        )
    return out


def encode_weights_legacy(weights: List[np.ndarray]) -> List[dict]:
    """Legacy format: [{"shape": [...], "data": [floats...]}]."""
    return [{"shape": list(w.shape), "data": w.astype(np.float32).ravel().tolist()} for w in weights]


def pack_blob(weights: List[np.ndarray]) -> str:
    """All tensors concatenated -> one base64 string (used for DB storage)."""
    flat = np.concatenate([np.ascontiguousarray(w, dtype="<f4").ravel() for w in weights])
    return base64.b64encode(flat.tobytes()).decode("ascii")


def unpack_blob(blob: str) -> List[np.ndarray]:
    """Inverse of pack_blob using the shapes from model_spec."""
    raw = base64.b64decode(blob)
    flat = np.frombuffer(raw, dtype="<f4")
    expected = model_spec.total_parameters()
    if flat.size != expected:
        raise PayloadError(f"stored weights have {flat.size} values, expected {expected}")
    out, pos = [], 0
    for spec in model_spec.LAYER_SPECS:
        n = int(np.prod(spec["shape"]))
        out.append(flat[pos : pos + n].reshape(spec["shape"]).astype(np.float32).copy())
        pos += n
    return out


# ── Decoding + validation (Privacy Enforcement Layer, server side) ───────

def _decode_tensor(tensor: Any, index: int) -> np.ndarray:
    field = f"weights[{index}]"
    if not isinstance(tensor, dict):
        raise PayloadError("each weight tensor must be an object with 'shape' and data", field)

    allowed = {"layer_name", "shape", "dtype", "data_b64", "data", "values"}
    extra = set(tensor.keys()) - allowed
    if extra:
        raise PayloadError(f"unexpected tensor fields: {sorted(extra)}", field)

    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape)
    ):
        raise PayloadError("tensor 'shape' must be a list of positive integers", field + ".shape")

    if "data_b64" in tensor:
        b64 = tensor["data_b64"]
        if not isinstance(b64, str):
            raise PayloadError("'data_b64' must be a base64 string", field + ".data_b64")
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception:
            raise PayloadError("'data_b64' is not valid base64", field + ".data_b64")
        if len(raw) % 4 != 0:
            raise PayloadError("'data_b64' is not a float32 buffer", field + ".data_b64")
        flat = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        data = tensor.get("data", tensor.get("values"))
        if not isinstance(data, list):
            raise PayloadError("tensor needs 'data_b64' (or legacy 'data' list)", field + ".data")
        flat_list = _flatten_numeric(data, field + ".data")
        flat = np.asarray(flat_list, dtype=np.float64)

    expected_count = int(np.prod(shape))
    if flat.size != expected_count:
        raise PayloadError(
            f"tensor has {flat.size} values but shape {shape} needs {expected_count}", field + ".data"
        )
    if not np.all(np.isfinite(flat)):
        raise PayloadError("tensor contains NaN or Infinity", field + ".data")
    return flat.astype(np.float32).reshape(shape)


def _flatten_numeric(lst, field):
    out = []
    stack = [lst]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(reversed(cur))
        elif isinstance(cur, bool) or not isinstance(cur, (int, float)):
            raise PayloadError("weights must contain numbers only (no text / booleans / objects)", field)
        else:
            if isinstance(cur, float) and (math.isnan(cur) or math.isinf(cur)):
                raise PayloadError("tensor contains NaN or Infinity", field)
            out.append(cur)
    return out


def decode_weights(payload: Any) -> List[np.ndarray]:
    """
    Validate + decode a client weight list into numpy arrays.
    Raises PayloadError (-> HTTP 400) on any violation, including shape
    mismatch against the global model architecture.
    """
    if not isinstance(payload, list) or not payload:
        raise PayloadError("'weights' must be a non-empty list of tensors", "weights")
    expected = model_spec.expected_shapes()
    if len(payload) != len(expected):
        raise PayloadError(
            f"expected {len(expected)} weight tensors, received {len(payload)}", "weights"
        )
    decoded = []
    for i, tensor in enumerate(payload):
        arr = _decode_tensor(tensor, i)
        if tuple(arr.shape) != expected[i]:
            raise PayloadError(
                f"tensor {i} has shape {list(arr.shape)}, expected {list(expected[i])}",
                f"weights[{i}].shape",
            )
        decoded.append(arr)
    return decoded
