"""
Synora: Browser-Based Federated Learning Toolkit
=================================================
Coordination Server — Single Entry Point

Implements all server-side REST API endpoints for the
federated learning coordination layer, covering:

    US-08  — Browser client registration
    US-11  — Aggregation participation threshold
    US-12  — Random client selection
    US-13  — Resource-aware client selection
    US-16  — FL experiment parameter configuration
    US-17  — Experiment results export (JSON + CSV)

Dependencies:
    pip install flask

Run:
    python coordination_server.py

Author: Kashaf Kamran
Project: SBFLT — Synora FYP
"""

from flask import Flask, request, jsonify, send_file
import uuid
import sys
import os

# ── Local module imports ──────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aggregation_config import AggregationConfig
from client_selector import ClientSelector
from experiment_tracker import ExperimentTracker

# ── App initialisation ────────────────────────────────────
app = Flask(__name__)


# ── In-memory state ───────────────────────────────────────
registered_clients = {}

# Available language pair partitions
# Each maps to a Kenyan low-resource language dataset
data_partitions = [
    "dav_swa",  # Kidawida — Kiswahili
    "kln_swa",  # Kalenjin — Kiswahili
    "luo_swa",  # Dholuo   — Kiswahili
]

# Aggregation configuration (min clients, max rounds)
config = AggregationConfig(min_clients=2, max_rounds=10)

# Client selector for random and resource-aware selection
selector = ClientSelector(selection_count=2)

# Experiment tracker for logging rounds and exporting results
tracker = ExperimentTracker(experiment_name="kenyan_fl_experiment")


# ── Helper functions ──────────────────────────────────────


def assign_partition(client_index: int) -> str:
    """
    Assign a language dataset partition to a client
    using round-robin distribution.

    Args:
        client_index (int): Current number of registered
                            clients before this one

    Returns:
        str: Partition identifier string
    """
    return data_partitions[client_index % len(data_partitions)]


# ═══════════════════════════════════════════════════════════
# US-08 — Client Registration
# ═══════════════════════════════════════════════════════════


@app.route("/register", methods=["POST"])
def register_client():
    """
    Register a browser client with the coordination server.

    Assigns a unique client ID and a language dataset
    partition. Rejects duplicate registrations.

    Request body (JSON):
        { "client_name": "client_A" }

    Returns:
        200 — Registration successful with client_id
              and partition assignment
        400 — Missing client_name field
        409 — Client already registered
    """
    data = request.get_json()

    if not data or "client_name" not in data:
        return jsonify({"error": "client_name is required"}), 400

    client_name = data["client_name"]

    # Reject duplicate registrations
    for cid, info in registered_clients.items():
        if info["client_name"] == client_name:
            return (
                jsonify(
                    {
                        "error": (f"Client '{client_name}' " f"is already registered"),
                        "client_id": cid,
                    }
                ),
                409,
            )

    # Generate unique client ID
    client_id = str(uuid.uuid4())

    # Assign partition using round-robin
    partition = assign_partition(len(registered_clients))

    # Store client record
    registered_clients[client_id] = {
        "client_name": client_name,
        "partition": partition,
        "status": "active",
    }

    return (
        jsonify(
            {
                "message": "Registration successful",
                "client_id": client_id,
                "partition": partition,
                "status": "active",
            }
        ),
        200,
    )


@app.route("/clients", methods=["GET"])
def get_clients():
    """
    Retrieve all registered clients and their details.

    Returns:
        200 — List of all clients with partition and status
    """
    return (
        jsonify(
            {"total_clients": len(registered_clients), "clients": registered_clients}
        ),
        200,
    )


# ═══════════════════════════════════════════════════════════
# US-11 — Aggregation Participation Threshold
# ═══════════════════════════════════════════════════════════


@app.route("/config/threshold", methods=["POST"])
def set_threshold():
    """
    Set the minimum number of clients required before
    aggregation is triggered for a federated round.

    Request body (JSON):
        { "min_clients": 3 }

    Returns:
        200 — Threshold updated with new config
        400 — Missing or invalid min_clients value
    """
    data = request.get_json()

    if not data or "min_clients" not in data:
        return jsonify({"error": "min_clients is required"}), 400

    try:
        min_clients = int(data["min_clients"])
        config.set_threshold(min_clients)
        return (
            jsonify(
                {
                    "message": f"Threshold set to {min_clients}",
                    "config": config.get_config(),
                }
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/config/threshold", methods=["GET"])
def get_threshold():
    """
    Retrieve the current aggregation threshold configuration.

    Returns:
        200 — Current min_clients and max_rounds config
    """
    return jsonify(config.get_config()), 200


@app.route("/aggregate/check", methods=["GET"])
def check_aggregation():
    """
    Check whether the current number of connected clients
    meets the aggregation threshold.

    Advances the round counter each time it is called.

    Returns:
        200 — Aggregation status with READY or WAITING
    """
    connected = len(registered_clients)
    can_aggregate = config.can_aggregate(connected)
    config.next_round()

    return (
        jsonify(
            {
                "can_aggregate": can_aggregate,
                "connected_clients": connected,
                "min_required": config.min_clients,
                "current_round": config.current_round,
                "status": ("READY" if can_aggregate else "WAITING — below threshold"),
            }
        ),
        200,
    )


# ═══════════════════════════════════════════════════════════
# US-12 + US-13 — Client Selection (Random + Resource-Aware)
# ═══════════════════════════════════════════════════════════


@app.route("/config/selection", methods=["POST"])
def set_selection_count():
    """
    Set the number of clients to select per federated round.

    Request body (JSON):
        { "selection_count": 2 }

    Returns:
        200 — Selection count updated
        400 — Missing or invalid selection_count
    """
    data = request.get_json()

    if not data or "selection_count" not in data:
        return jsonify({"error": "selection_count is required"}), 400

    try:
        count = int(data["selection_count"])
        selector.set_selection_count(count)
        return (
            jsonify(
                {"message": f"Selection count set to {count}", "selection_count": count}
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/select/clients", methods=["GET"])
def select_clients():
    """
    Select a subset of registered clients to participate
    in the current federated training round.

    Uses the ClientSelector strategy (random or
    resource-aware depending on configuration).

    Returns:
        200 — Selected clients with their partition details
        400 — No clients registered
    """
    available = list(registered_clients.keys())

    if len(available) == 0:
        return jsonify({"error": "No clients registered"}), 400

    selected = selector.select_clients(available)
    selected_info = {cid: registered_clients[cid] for cid in selected}

    return (
        jsonify(
            {
                "round": selector.round_number,
                "selected_count": len(selected),
                "selected_clients": selected_info,
                "total_available": len(available),
            }
        ),
        200,
    )


@app.route("/select/history", methods=["GET"])
def get_selection_history():
    """
    Retrieve client selection history including frequency
    distribution and bias analysis across all rounds.

    Returns:
        200 — Stats, frequency distribution, bias check
    """
    available = list(registered_clients.keys())

    return (
        jsonify(
            {
                "stats": selector.get_stats(),
                "frequency_distribution": (
                    selector.get_frequency_distribution(available)
                ),
                "bias_check": selector.check_bias(available),
            }
        ),
        200,
    )


# ═══════════════════════════════════════════════════════════
# US-16 — FL Experiment Parameter Configuration
# ═══════════════════════════════════════════════════════════


@app.route("/experiment/config", methods=["POST"])
def set_experiment_config():
    """
    Save the FL experiment hyperparameter configuration.

    Request body (JSON):
        {
            "num_rounds": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 5,
            "partition_type": "non_iid",
            "dirichlet_alpha": 0.5
        }

    Returns:
        200 — Config saved successfully
        400 — No config data provided
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Config data required"}), 400

    tracker.set_config(data)

    return jsonify({"message": "Experiment config saved", "config": data}), 200


# ═══════════════════════════════════════════════════════════
# US-17 — Experiment Results Export
# ═══════════════════════════════════════════════════════════


@app.route("/experiment/log", methods=["POST"])
def log_round():
    """
    Log metrics for a completed federated training round.

    Request body (JSON):
        {
            "round": 1,
            "accuracy": 0.72,
            "loss": 0.54,
            "participating_clients": ["id1", "id2"]
        }

    Returns:
        200 — Round logged with stored data
        400 — Missing or invalid round data
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Round data required"}), 400

    try:
        round_data = tracker.log_round(
            round_num=data.get("round", len(tracker.rounds) + 1),
            accuracy=float(data.get("accuracy", 0)),
            loss=float(data.get("loss", 0)),
            participating_clients=data.get("participating_clients", []),
        )
        return jsonify({"message": "Round logged", "round_data": round_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/summary", methods=["GET"])
def get_summary():
    """
    Retrieve a full summary of the experiment including
    all round metrics, accuracy trend, and configuration.

    Returns:
        200 — Complete experiment summary
    """
    return jsonify(tracker.get_summary()), 200


@app.route("/experiment/export/json", methods=["GET"])
def export_json():
    """
    Export all experiment results as a downloadable
    JSON file for external analysis.

    Returns:
        200 — JSON file download
        400 — Export error
    """
    try:
        filepath = os.path.join(
            os.path.dirname(__file__), "kenyan_fl_experiment_results.json"
        )
        tracker.export_json(filepath)
        return send_file(
            filepath, as_attachment=True, download_name="experiment_results.json"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/export/csv", methods=["GET"])
def export_csv():
    """
    Export all experiment results as a downloadable
    CSV file for spreadsheet analysis.

    Returns:
        200 — CSV file download
        400 — Export error
    """
    try:
        filepath = os.path.join(
            os.path.dirname(__file__), "kenyan_fl_experiment_results.csv"
        )
        tracker.export_csv(filepath)
        return send_file(
            filepath, as_attachment=True, download_name="experiment_results.csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════
# Server Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Synora Coordination Server")
    print("  Browser-Based Federated Learning Toolkit")
    print("=" * 55)
    print()
    print("  CLIENT REGISTRATION (US-08)")
    print("  POST /register")
    print("  GET  /clients")
    print()
    print("  AGGREGATION THRESHOLD (US-11)")
    print("  POST /config/threshold")
    print("  GET  /config/threshold")
    print("  GET  /aggregate/check")
    print()
    print("  CLIENT SELECTION (US-12, US-13)")
    print("  POST /config/selection")
    print("  GET  /select/clients")
    print("  GET  /select/history")
    print()
    print("  EXPERIMENT CONFIG (US-16)")
    print("  POST /experiment/config")
    print()
    print("  RESULTS EXPORT (US-17)")
    print("  POST /experiment/log")
    print("  GET  /experiment/summary")
    print("  GET  /experiment/export/json")
    print("  GET  /experiment/export/csv")
    print()
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)

    app.run(debug=True)
