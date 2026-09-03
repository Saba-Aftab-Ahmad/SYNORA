"""
Synora: Browser-Based Federated Learning Toolkit
=================================================
Coordination Server — Production Version

All state is persisted in PostgreSQL via Supabase.
No in-memory storage — safe to restart at any time.

User Stories covered:
    US-08  — Browser client registration
    US-11  — Aggregation participation threshold
    US-12  — Random client selection
    US-13  — Resource-aware client selection
    US-16  — FL experiment parameter configuration
    US-17  — Experiment results export (JSON + CSV)

Run locally:
    python server/coordination_server.py

Run in production:
    gunicorn server.coordination_server:app

Author: Kashaf Kamran
Project: SBFLT — Synora FYP
"""

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
    AggregationConfig
)

# ── App initialisation ────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Initialise database
init_db(app)
migrate = Migrate(app, db)

# Available language pair partitions
DATA_PARTITIONS = ["dav_swa", "kln_swa", "luo_swa"]


def assign_partition(client_count: int) -> str:
    """Assign partition using round-robin distribution."""
    return DATA_PARTITIONS[client_count % len(DATA_PARTITIONS)]


def get_or_create_aggregation_config():
    """Get aggregation config from DB or create default."""
    config = AggregationConfig.query.first()
    if not config:
        config = AggregationConfig(
            min_clients=2,
            max_rounds=10,
            current_round=0
        )
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
        return jsonify({
            "error": f"Client '{client_name}' is already registered",
            "client_id": existing.client_id
        }), 409

    client_id = str(uuid.uuid4())
    client_count = Client.query.count()
    partition = assign_partition(client_count)

    new_client = Client(
        client_id=client_id,
        client_name=client_name,
        partition=partition,
        status="active"
    )

    db.session.add(new_client)
    db.session.commit()

    return jsonify({
        "message": "Registration successful",
        "client_id": client_id,
        "partition": partition,
        "status": "active"
    }), 200


@app.route("/clients", methods=["GET"])
def get_clients():
    clients = Client.query.all()
    return jsonify({
        "total_clients": len(clients),
        "clients": {c.client_id: c.to_dict() for c in clients}
    }), 200


@app.route("/clients/reset", methods=["DELETE"])
def reset_clients():
    Client.query.delete()
    db.session.commit()
    return jsonify({"message": "All clients cleared"}), 200


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

        return jsonify({
            "message": f"Threshold set to {min_clients}",
            "config": config.to_dict()
        }), 200

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

    return jsonify({
        "can_aggregate": can_aggregate,
        "connected_clients": connected,
        "min_required": config.min_clients,
        "current_round": config.current_round,
        "status": "READY" if can_aggregate else "WAITING — below threshold"
    }), 200


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

        return jsonify({
            "message": f"Selection count set to {count}",
            "selection_count": count
        }), 200

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

    return jsonify({
        "round": config.current_round,
        "selected_count": len(selected),
        "selected_clients": {c.client_id: c.to_dict() for c in selected},
        "total_available": len(all_clients)
    }), 200


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

    return jsonify({
        "total_rounds": total_rounds,
        "participation_counts": participation,
        "participation_rates": {
            name: (round(count / total_rounds, 4) if total_rounds > 0 else 0)
            for name, count in participation.items()
        }
    }), 200


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
        languages=",".join(data.get("languages", ["dholuo", "kalenjin", "kidawida"]))
    )

    db.session.add(config)
    db.session.commit()

    return jsonify({
        "message": "Experiment config saved",
        "config": config.to_dict()
    }), 200


@app.route("/experiment/config", methods=["GET"])
def get_experiment_config():
    config = ExperimentConfig.query.order_by(
        ExperimentConfig.created_at.desc()
    ).first()

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
            client_count=len(participating)
        )

        db.session.add(log)
        db.session.commit()

        return jsonify({
            "message": "Round logged",
            "round_data": log.to_dict()
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/summary", methods=["GET"])
def get_summary():
    logs = ExperimentLog.query.order_by(ExperimentLog.round_number).all()
    config = ExperimentConfig.query.order_by(
        ExperimentConfig.created_at.desc()
    ).first()

    return jsonify({
        "experiment_name": config.experiment_name if config else "kenyan_fl_experiment",
        "total_rounds": len(logs),
        "configuration": config.to_dict() if config else {},
        "rounds": [log.to_dict() for log in logs],
        "start_time": logs[0].logged_at.isoformat() if logs else None,
        "export_time": datetime.utcnow().isoformat()
    }), 200


@app.route("/experiment/export/json", methods=["GET"])
def export_json():
    try:
        logs = ExperimentLog.query.order_by(ExperimentLog.round_number).all()
        config = ExperimentConfig.query.order_by(
            ExperimentConfig.created_at.desc()
        ).first()

        payload = {
            "experiment_name": config.experiment_name if config else "kenyan_fl_experiment",
            "export_time": datetime.utcnow().isoformat(),
            "configuration": config.to_dict() if config else {},
            "total_rounds": len(logs),
            "rounds": [log.to_dict() for log in logs]
        }

        buffer = io.BytesIO()
        buffer.write(json.dumps(payload, indent=2).encode("utf-8"))
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/json",
            as_attachment=True,
            download_name="experiment_results.json"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/experiment/export/csv", methods=["GET"])
def export_csv():
    try:
        logs = ExperimentLog.query.order_by(ExperimentLog.round_number).all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "round", "accuracy", "loss",
            "client_count", "participating_clients", "logged_at"
        ])

        for log in logs:
            writer.writerow([
                log.round_number, log.accuracy, log.loss,
                log.client_count, log.participating_clients,
                log.logged_at.isoformat()
            ])

        buffer = io.BytesIO()
        buffer.write(output.getvalue().encode("utf-8"))
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name="experiment_results.csv"
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
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "registered_clients": client_count,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


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
