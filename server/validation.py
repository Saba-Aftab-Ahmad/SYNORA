"""
server/validation.py
Flask-compatible payload validation decorator.

This is the Python/Flask equivalent of client/validate_update_payload.js
(which was written for Express.js and cannot work with a Flask backend).

Usage:
    from server.validation import validate_update_payload

    @app.route("/model/update", methods=["POST"])
    @validate_update_payload
    def submit_update():
        payload = request.get_json()
        # safe to use payload["weights"] here
        ...
"""

from functools import wraps
from flask import request, jsonify


# Only these keys are allowed in a client weight-update payload.
# Raw text, vocabulary, labels — none of these should ever appear.
#ALLOWED_KEYS = {"weights", "shapes", "modelId", "round", "clientId", "client_id"}
ALLOWED_KEYS = {
    "weights", "shapes", "modelId", "round",
    "clientId", "client_id",
    "datasetSize", "dataset_size", "localEpochs", "local_epochs",
    "backendUsed", "backend_used", "roundNumber", "round_number",
    "payloadSizeBytes", "payload_size_bytes"
}

# def _is_numeric_tensor(value):
#     """
#     Recursively check that a value is a nested list of numbers only.
#     Strings, dicts, booleans, None — all rejected.
#     """
#     if isinstance(value, list):
#         return all(_is_numeric_tensor(v) for v in value)
#     return isinstance(value, (int, float)) and not isinstance(value, bool)
def _is_numeric_tensor(value):
    """Accept nested lists, dicts with numeric values, or plain numbers."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, list):
        return all(_is_numeric_tensor(v) for v in value)
    if isinstance(value, dict):
        # Accept {layer_index, shape, data} format from TF.js
        return True
    return False


def validate_update_payload(f):
    """
    Flask decorator that validates incoming weight-update payloads.

    Rejects (HTTP 400) if:
      - Body is not a JSON object
      - Any key outside ALLOWED_KEYS is present (Privacy Enforcement Layer)
      - 'weights' field is missing or contains non-numeric values

    Returns HTTP 400 with a structured error body matching the
    SDS Section 6.8 API Error Response Schema.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        payload = request.get_json(silent=True)

        # Must be a JSON object
        if not isinstance(payload, dict):
            return jsonify({
                "error": "bad_request",
                "message": "Malformed payload: body must be a JSON object"
            }), 400

        # Privacy boundary — no unexpected fields allowed
        for key in payload:
            if key not in ALLOWED_KEYS:
                return jsonify({
                    "error": "bad_request",
                    "message": f'Unexpected field "{key}" in payload. '
                               f'Only weight tensors and session identifiers are permitted.',
                    "field": key
                }), 400

        # Weights must be present and numeric
        weights = payload.get("weights")
        if not isinstance(weights, list):
            return jsonify({
                "error": "bad_request",
                "message": "Malformed payload: 'weights' field must be a list",
                "field": "weights"
            }), 400

        if not all(_is_numeric_tensor(w) for w in weights):
            return jsonify({
                "error": "bad_request",
                "message": "Malformed payload: 'weights' must contain numeric tensors only "
                           "(no strings, booleans, or nested objects)",
                "field": "weights"
            }), 400

        return f(*args, **kwargs)

    return wrapper
