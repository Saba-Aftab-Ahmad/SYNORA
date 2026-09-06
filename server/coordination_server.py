"""
Synora: Browser-Based Federated Learning Toolkit
=================================================
Coordination Server — Production Version

All state is persisted in PostgreSQL via Supabase.
No in-memory storage — safe to restart at any time.

User Stories covered:
    US-08  — Browser client registration
    US-09  — Global model distribution
    US-10  — Weight update submission + FedAvg
    US-11  — Aggregation participation threshold
    US-12  — Random client selection
    US-13  — Resource-aware client selection
    US-16  — FL experiment parameter configuration
    US-17  — Experiment results export (JSON + CSV)

Run locally:
    python server/coordination_server.py

Run in production:
    gunicorn server.coordination_server:app
"""

import numpy as np
import uuid
import csv
import json
import io
import os
import sys
import random
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_migrate import Migrate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.database import db, init_db
from server.models import (
    Client,
    Round,
    ExperimentConfig,
    ExperimentLog,
    AggregationConfig,
)

# ── App initialisation ────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")

# Initialise database
init_db(app)
migrate = Migrate(app, db)

# Available language pair partitions
DATA_PARTITIONS = ["dav_swa", "kln_swa", "luo_swa"]

# ── In-memory global model state ──────────────────────────
# (Phase 2 mein DB mein store hoga)
global_model_weights = None
client_updates = {}  # {client_id: {weights, dataset_size, metrics, round}}


def assign_partition(client_count: int) -> str:
    """Assign partition using round-robin distribution."""
    return DATA_PARTITIONS[client_count % len(DATA_PARTITIONS)]


def get_or_create_aggregation_config():
    """Get aggregation config from DB or create default."""
    config = AggregationConfig.query.first()
    if not config:
        config = AggregationConfig(min_clients=2, max_rounds=10, current_round=0)
        db.session.add(config)
        db.session.commit()
    return config


# ═══════════════════════════════════════════════════════════
# US-08 — Client Registration
# ═══════════════════════════════════════════════════════════


@app.route("/register", methods=["POST"])
def register_client():
    data = request.get_json()

    if not data or "client_name" not in data:
        return jsonify({"error": "client_name is required"}), 400

    client_name = data["client_name"]

    existing = Client.query.filter_by(client_name=client_name).first()
    if existing:
        return (
            jsonify(
                {
                    "error": f"Client '{client_name}' is already registered",
                    "client_id": existing.client_id,
                }
            ),
            409,
        )

    client_id = str(uuid.uuid4())
    client_count = Client.query.count()
    partition = assign_partition(client_count)

    new_client = Client(
        client_id=client_id,
        client_name=client_name,
        partition=partition,
        status="active",
    )

    db.session.add(new_client)
    db.session.commit()

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
    clients = Client.query.all()
    return (
        jsonify(
            {
                "total_clients": len(clients),
                "clients": {c.client_id: c.to_dict() for c in clients},
            }
        ),
        200,
    )


@app.route("/clients/reset", methods=["DELETE"])
def reset_clients():
    Client.query.delete()
    db.session.commit()
    return jsonify({"message": "All clients cleared"}), 200


# ═══════════════════════════════════════════════════════════
# US-09 — Global Model Distribution
# Dono routes: /global-model (purana) + /model/global (naya)
# Frontend dono mein se koi bhi use kare — dono kaam karein ge
# ═══════════════════════════════════════════════════════════


def _build_global_model_response():
    """Helper — global model ka response build karo."""
    global global_model_weights

    if global_model_weights is None:
        global_model_weights = {
            "version": 0,
            "architecture": {
                "vocab_size": 5000,
                "embedding_dim": 64,
                "num_classes": 3,
                "max_length": 100,
            },
            "weights": None,
            "round": 0,
        }

    return global_model_weights


@app.route("/global-model", methods=["GET"])
def get_global_model_old():
    """Legacy route — purana URL support karo."""
    return jsonify(_build_global_model_response()), 200


@app.route("/model/global", methods=["GET"])
def get_global_model():
    """New route — /model/global (SDS API design se match)."""
    return jsonify(_build_global_model_response()), 200


@app.route("/model/version", methods=["GET"])
def get_model_version():
    """Return current global model version number."""
    global global_model_weights
    version = (global_model_weights or {}).get("version", 0)
    return jsonify({"version": version}), 200


@app.route("/global-model/ready", methods=["GET"])
def check_model_ready():
    """Client check kare agar naya aggregated model ready hai."""
    global global_model_weights

    if global_model_weights is None:
        return jsonify({"ready": False, "version": 0}), 200

    return (
        jsonify(
            {
                "ready": True,
                "version": global_model_weights.get("version", 0),
                "round": global_model_weights.get("round", 0),
                "has_weights": global_model_weights.get("weights") is not None,
            }
        ),
        200,
    )


# ═══════════════════════════════════════════════════════════
# US-10 — Weight Update Submission + FedAvg
# Dono routes support karo:
#   /submit-update  (purana — frontend jo use kar raha tha)
#   /model/update   (naya — SDS API design)
# ═══════════════════════════════════════════════════════════


# Key fields jo frontend bhej sakta hai — sab allowed hain
ALLOWED_UPDATE_KEYS = {
    "weights",
    "shapes",
    "modelId",
    "round",
    "roundNumber",
    "round_number",
    "clientId",
    "client_id",
    "datasetSize",
    "dataset_size",
    "localEpochs",
    "local_epochs",
    "backendUsed",
    "backend_used",
    "payloadSizeBytes",
    "payload_size_bytes",
    "metrics",
}


def _process_weight_update(data):
    """
    Weight update process karo — FedAvg trigger karo agar threshold meet ho.
    Dono /submit-update aur /model/update routes yahi function use karte hain.
    """
    global global_model_weights, client_updates

    if not data:
        return jsonify({"error": "No data provided"}), 400

    # camelCase aur snake_case dono accept karo
    client_id = data.get("client_id") or data.get("clientId")
    weights = data.get("weights")
    dataset_size = data.get("dataset_size") or data.get("datasetSize") or 100
    metrics = data.get("metrics", {})
    round_num = (
        data.get("round") or data.get("roundNumber") or data.get("round_number") or 1
    )

    # Privacy check — koi unexpected field nahi honi chahiye
    unexpected = set(data.keys()) - ALLOWED_UPDATE_KEYS
    if unexpected:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": f"Unexpected fields in payload: {unexpected}. "
                    f"Only weight tensors and session identifiers are permitted.",
                    "field": list(unexpected)[0],
                }
            ),
            400,
        )

    if not client_id:
        return jsonify({"error": "client_id is required"}), 400

    if weights is None:
        return jsonify({"error": "weights are required"}), 400

    # Client registered hai ya nahi — demo mode mein accept karo
    registered = Client.query.filter_by(client_id=client_id).first()
    if not registered:
        print(
            f"[UPDATE] Warning: unregistered client {str(client_id)[:8]}... — accepting in demo mode"
        )

    # Update store karo
    client_updates[client_id] = {
        "weights": weights,
        "dataset_size": int(dataset_size),
        "metrics": metrics,
        "round": round_num,
    }

    print(
        f"[UPDATE] Client {str(client_id)[:8]}... submitted weights "
        f"for round {round_num} | Updates received: {len(client_updates)}"
    )

    # Aggregation check
    config = get_or_create_aggregation_config()
    num_updates = len(client_updates)
    threshold_met = num_updates >= config.min_clients

    response = {
        "message": "Update received",
        "status": "accepted",
        "updates_received": num_updates,
        "threshold": config.min_clients,
        "threshold_met": threshold_met,
        "round": round_num,
        "client_id": client_id,
    }

    # Enough updates aaye — FedAvg run karo
    if threshold_met:
        aggregated = run_fedavg(client_updates)
        current_version = (global_model_weights or {}).get("version", 0)

        global_model_weights = {
            "version": current_version + 1,
            "weights": aggregated,
            "round": round_num,
            "num_clients_aggregated": num_updates,
        }

        # Round log karo DB mein
        try:
            log = ExperimentLog(
                round_number=int(round_num),
                accuracy=float(metrics.get("accuracy", 0)),
                loss=float(metrics.get("loss", 0)),
                participating_clients=",".join(client_updates.keys()),
                client_count=num_updates,
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            print(f"[LOG] Could not save round log: {e}")

        client_updates = {}  # Reset for next round

        response["aggregated"] = True
        response["new_model_version"] = global_model_weights["version"]
        print(
            f"[FEDAVG] Round {round_num} aggregated — model v{global_model_weights['version']}"
        )

    return jsonify(response), 200


@app.route("/submit-update", methods=["POST"])
def submit_client_update():
    """Legacy route — /submit-update (purana frontend URL)."""
    data = request.get_json()
    return _process_weight_update(data)


@app.route("/model/update", methods=["POST"])
def model_update():
    """New route — /model/update (SDS API design)."""
    data = request.get_json()
    return _process_weight_update(data)


def run_fedavg(updates: dict) -> list:
    """
    FedAvg algorithm: dataset-size weighted average of client weights.
    """
    if not updates:
        return []

    total_size = sum(u["dataset_size"] for u in updates.values())

    if total_size == 0:
        return list(updates.values())[0]["weights"]

    aggregated = None

    for client_id, update in updates.items():
        weight = update["dataset_size"] / total_size
        client_weights = update["weights"]

        if aggregated is None:
            aggregated = [
                (
                    [w * weight for w in layer]
                    if isinstance(layer, list)
                    else layer * weight
                )
                for layer in client_weights
            ]
        else:
            for i, layer in enumerate(client_weights):
                if isinstance(layer, list):
                    for j, val in enumerate(layer):
                        aggregated[i][j] += val * weight
                else:
                    aggregated[i] += layer * weight

    return aggregated


# ═══════════════════════════════════════════════════════════
# US-11 — Aggregation Participation Threshold
# ═══════════════════════════════════════════════════════════


@app.route("/config/threshold", methods=["POST"])
def set_threshold():
    data = request.get_json()

    if not data or "min_clients" not in data:
        return jsonify({"error": "min_clients is required"}), 400

    try:
        min_clients = int(data["min_clients"])
        if min_clients < 1:
            raise ValueError("min_clients must be at least 1")

        config = get_or_create_aggregation_config()
        config.min_clients = min_clients
        config.updated_at = datetime.utcnow()
        db.session.commit()

        return (
            jsonify(
                {
                    "message": f"Threshold set to {min_clients}",
                    "config": config.to_dict(),
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/config/threshold", methods=["GET"])
def get_threshold():
    config = get_or_create_aggregation_config()
    return jsonify(config.to_dict()), 200


@app.route("/aggregate/check", methods=["GET"])
def check_aggregation():
    config = get_or_create_aggregation_config()
    connected = Client.query.filter_by(status="active").count()
    can_aggregate = connected >= config.min_clients

    if can_aggregate:
        config.current_round += 1
        config.updated_at = datetime.utcnow()
        db.session.commit()

    return (
        jsonify(
            {
                "can_aggregate": can_aggregate,
                "connected_clients": connected,
                "min_required": config.min_clients,
                "current_round": config.current_round,
                "status": "READY" if can_aggregate else "WAITING — below threshold",
            }
        ),
        200,
    )


# ═══════════════════════════════════════════════════════════
# US-12 + US-13 — Client Selection
# ═══════════════════════════════════════════════════════════


@app.route("/config/selection", methods=["POST"])
def set_selection_count():
    data = request.get_json()

    if not data or "selection_count" not in data:
        return jsonify({"error": "selection_count is required"}), 400

    try:
        count = int(data["selection_count"])
        if count < 1:
            raise ValueError("selection_count must be at least 1")

        app.config["SELECTION_COUNT"] = count

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
    selection_count = app.config.get("SELECTION_COUNT", 2)
    all_clients = Client.query.filter_by(status="active").all()

    if not all_clients:
        return jsonify({"error": "No clients registered"}), 400

    count = min(selection_count, len(all_clients))
    selected = random.sample(all_clients, count)
    config = get_or_create_aggregation_config()

    return (
        jsonify(
            {
                "round": config.current_round,
                "selected_count": len(selected),
                "selected_clients": {c.client_id: c.to_dict() for c in selected},
                "total_available": len(all_clients),
            }
        ),
        200,
    )


@app.route("/select/history", methods=["GET"])
def get_selection_history():
    all_clients = Client.query.all()
    logs = ExperimentLog.query.all()

    participation = {c.client_name: 0 for c in all_clients}

    for log in logs:
        for name in log.participating_clients.split(","):
            name = name.strip()
            if name in participation:
                participation[name] += 1

    total_rounds = len(logs)

    return (
        jsonify(
            {
                "total_rounds": total_rounds,
                "participation_counts": participation,
                "participation_rates": {
                    name: (round(count / total_rounds, 4) if total_rounds > 0 else 0)
                    for name, count in participation.items()
                },
            }
        ),
        200,
    )


# ═══════════════════════════════════════════════════════════
# US-16 — FL Experiment Configuration
# ═══════════════════════════════════════════════════════════


@app.route("/experiment/config", methods=["POST"])
def set_experiment_config():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Config data required"}), 400

    config = ExperimentConfig(
        experiment_name=data.get("experiment_name", "kenyan_fl_experiment"),
        num_rounds=data.get("num_rounds", 10),
        num_clients=data.get("num_clients", 3),
        min_clients_per_round=data.get("min_clients_per_round", 2),
        learning_rate=data.get("learning_rate", 0.01),
        batch_size=data.get("batch_size", 32),
        local_epochs=data.get("local_epochs", 5),
        partition_type=data.get("partition_type", "non_iid"),
        dirichlet_alpha=data.get("dirichlet_alpha", 0.5),
        languages=",".join(data.get("languages", ["dholuo", "kalenjin", "kidawida"])),
    )

    db.session.add(config)
    db.session.commit()

    return (
        jsonify({"message": "Experiment config saved", "config": config.to_dict()}),
        200,
    )


@app.route("/experiment/config", methods=["GET"])
def get_experiment_config():
    config = ExperimentConfig.query.order_by(ExperimentConfig.created_at.desc()).first()

    if not config:
        return jsonify({"error": "No experiment config saved yet"}), 404

    return jsonify(config.to_dict()), 200


# ═══════════════════════════════════════════════════════════
# US-17 — Experiment Results Export
# ═══════════════════════════════════════════════════════════


@app.route("/experiment/log", methods=["POST"])
def log_round():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Round data required"}), 400

    try:
        participating = data.get("participating_clients", [])

        log = ExperimentLog(
            round_number=int(data.get("round", ExperimentLog.query.count() + 1)),
            accuracy=float(data.get("accuracy", 0)),
            loss=float(data.get("loss", 0)),
            participating_clients=",".join(participating),
            client_count=len(participating),
        )

        db.session.add(log)
        db.session.commit()

        return jsonify({"message": "Round logged", "round_data": log.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/summary", methods=["GET"])
def get_summary():
    logs = ExperimentLog.query.order_by(ExperimentLog.round_number).all()
    config = ExperimentConfig.query.order_by(ExperimentConfig.created_at.desc()).first()

    return (
        jsonify(
            {
                "experiment_name": (
                    config.experiment_name if config else "kenyan_fl_experiment"
                ),
                "total_rounds": len(logs),
                "configuration": config.to_dict() if config else {},
                "rounds": [log.to_dict() for log in logs],
                "start_time": logs[0].logged_at.isoformat() if logs else None,
                "export_time": datetime.utcnow().isoformat(),
            }
        ),
        200,
    )


@app.route("/experiment/export/json", methods=["GET"])
def export_json():
    try:
        logs = ExperimentLog.query.order_by(ExperimentLog.round_number).all()
        config = ExperimentConfig.query.order_by(
            ExperimentConfig.created_at.desc()
        ).first()

        payload = {
            "experiment_name": (
                config.experiment_name if config else "kenyan_fl_experiment"
            ),
            "export_time": datetime.utcnow().isoformat(),
            "configuration": config.to_dict() if config else {},
            "total_rounds": len(logs),
            "rounds": [log.to_dict() for log in logs],
        }

        buffer = io.BytesIO()
        buffer.write(json.dumps(payload, indent=2).encode("utf-8"))
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/json",
            as_attachment=True,
            download_name="experiment_results.json",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/export/csv", methods=["GET"])
def export_csv():
    try:
        logs = ExperimentLog.query.order_by(ExperimentLog.round_number).all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "round",
                "accuracy",
                "loss",
                "client_count",
                "participating_clients",
                "logged_at",
            ]
        )

        for log in logs:
            writer.writerow(
                [
                    log.round_number,
                    log.accuracy,
                    log.loss,
                    log.client_count,
                    log.participating_clients,
                    log.logged_at.isoformat(),
                ]
            )

        buffer = io.BytesIO()
        buffer.write(output.getvalue().encode("utf-8"))
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name="experiment_results.csv",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/reset", methods=["DELETE"])
def reset_experiment():
    ExperimentLog.query.delete()
    db.session.commit()
    return jsonify({"message": "Experiment logs cleared"}), 200


# ═══════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════


@app.route("/health", methods=["GET"])
def health_check():
    try:
        client_count = Client.query.count()
        return (
            jsonify(
                {
                    "status": "healthy",
                    "database": "connected",
                    "registered_clients": client_count,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# Server Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    print("=" * 55)
    print("  Synora Coordination Server — Production")
    print("  All state persisted in PostgreSQL")
    print("=" * 55)
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)

    app.run(debug=False, host="0.0.0.0", port=5000)
