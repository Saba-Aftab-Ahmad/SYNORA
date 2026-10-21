"""
Synora Coordination Server  (v2)
================================
Flask application that coordinates federated training between browser clients.

  * /api/...        - the documented REST contract (sessions, clients, rounds,
                      model, metrics, selection, export).
  * legacy routes   - /register, /clients, /global-model, /submit-update, ...
                      kept as thin wrappers so older tools / tests still work.

All coordination state is stored in the database (see fl_service.py), so the
server is correct with several gunicorn workers and recovers after a restart.
"""

import csv
import io
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from functools import wraps

from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS
from flask_migrate import Migrate
from sqlalchemy import text

# make `python server/coordination_server.py` work too (project root on the path)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server import data_service, fl_service, model_spec, weight_codec
from server.api_errors import ApiError, bad_request, forbidden, not_found, unauthorized
from server.database import db, ensure_schema, init_db
from server.models import (
    AggregationConfig,
    Client,
    ClientEvent,
    ClientSelectionLog,
    ExperimentConfig,
    ExperimentLog,
    ExperimentSession,
    Round,
    iso,
    utcnow,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
CORS(
    app,
    origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_headers=["Content-Type", "X-Session-Token", "X-Admin-Key"],
    expose_headers=["Content-Disposition"],
)

init_db(app)
migrate = Migrate(app, db)

# Create missing tables / add missing columns (works under gunicorn too; the
# old code only called db.create_all() under `if __name__ == "__main__"`).
try:
    ensure_schema(app)
except Exception as _exc:  # pragma: no cover
    print(f"[schema] WARNING: could not verify schema at start-up: {_exc}")


# ── cross-cutting: errors, security headers, auth ────────────────────────

@app.errorhandler(ApiError)
def _handle_api_error(e: ApiError):
    db.session.rollback()
    body, status = e.to_response()
    return jsonify(body), status


@app.errorhandler(404)
def _handle_404(e):
    return jsonify({"error": "not_found", "message": "Resource not found", "field": None, "reference_id": str(uuid.uuid4())}), 404


@app.errorhandler(405)
def _handle_405(e):
    return jsonify({"error": "method_not_allowed", "message": "Method not allowed", "field": None, "reference_id": str(uuid.uuid4())}), 405


@app.errorhandler(413)
def _handle_413(e):
    return jsonify({"error": "payload_too_large", "message": "Request body too large", "field": None, "reference_id": str(uuid.uuid4())}), 413


@app.errorhandler(Exception)
def _handle_500(e):
    db.session.rollback()
    ref = str(uuid.uuid4())
    print(f"[ERROR {ref}] {request.method} {request.path}\n{traceback.format_exc()}")
    return jsonify({"error": "internal_error", "message": "Unexpected server error", "field": None, "reference_id": ref}), 500


@app.after_request
def _security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    if request.headers.get("X-Forwarded-Proto", request.scheme) == "https":
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


def json_body():
    data = request.get_json(silent=True)
    if data is None:
        raise bad_request("A JSON request body is required")
    return data


def require_client(f):
    """Session-token authentication (SDS: 401 on missing/invalid token)."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-Session-Token")
        if not token and request.method == "POST" and request.is_json:
            token = (request.get_json(silent=True) or {}).get("session_token")  # sendBeacon cannot set headers
        if not token:
            raise unauthorized()
        client = Client.query.filter_by(session_token=token).first()
        if client is None or client.status == "left":
            raise unauthorized("Session token is invalid or the client has left the session")
        g.client = client
        return f(*args, **kwargs)

    return wrapper


def require_admin(f):
    """Destructive maintenance endpoints need X-Admin-Key == ADMIN_KEY env var."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        key = os.getenv("ADMIN_KEY")
        if not key:
            raise forbidden("Disabled: set the ADMIN_KEY environment variable to enable this endpoint", "admin_disabled")
        if request.headers.get("X-Admin-Key") != key:
            raise forbidden("Invalid admin key", "admin_forbidden")
        return f(*args, **kwargs)

    return wrapper


def _client_public(client: Client, include_token=False):
    d = client.to_dict()
    if include_token:
        d["session_token"] = client.session_token
    return d


def _session_for_metrics():
    sid = request.args.get("session_id")
    if sid:
        sess = fl_service.get_session(sid)
        if sess is None:
            raise not_found("Session not found", "session_not_found")
        return sess
    return fl_service.latest_session()


# ═════════════════════════════════════════════════════════════════════════
# /api  — sessions & clients (US-06, US-08, US-12/13, US-16)
# ═════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def api_health():
    try:
        db.session.execute(text("SELECT 1"))
        return jsonify({"status": "healthy", "database": "connected", "timestamp": iso(utcnow()), "version": "2.0"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/api/clients/join", methods=["POST"])
def api_join():
    """
    Register this browser (with its hardware capabilities) and take part in the
    running training - or start one when none is running.  Nothing is waited for:
    one client is enough, and any number can join later.
    Body: {client_name?, config?: {total_rounds?, partition_type?, dirichlet_alpha?},
           capabilities: {backend, cpu_cores, memory_gb, webgpu_available,
           webnn_available}, preferred_language?,
           mode?: "auto" | "join" | "new", session_id?}
      auto (default) - join the running training, else start one from `config`
      join           - join the running session `session_id`
      new            - start a separate session even if one is running (API/tests)
    Learning rate, epochs, batch size, timeout and client selection are fixed on the
    server (server/training_settings.py); `config` can only set rounds + data split.
    """
    body = json_body()
    caps = body.get("capabilities") or {}
    info = {
        "client_name": body.get("client_name"),
        "preferred_language": body.get("preferred_language"),
        "user_agent": request.headers.get("User-Agent"),
        **{k: caps.get(k) for k in ("backend", "cpu_cores", "memory_gb", "webgpu_available", "webnn_available")},
    }
    mode = body.get("mode") or "auto"
    sid = body.get("session_id")
    if sid is not None and not isinstance(sid, str):
        raise bad_request("session_id must be a string", "session_id")
    result = fl_service.join_session(body.get("config"), info, mode=mode, session_id=sid)
    sess, client = result["session"], result["client"]
    return (
        jsonify(
            {
                "message": "Registration successful",
                "created_session": result["created_session"],
                "client": _client_public(client, include_token=True),
                "session": sess.to_dict(),
                "architecture": model_spec.architecture(),
            }
        ),
        200,
    )


@app.route("/api/clients/leave", methods=["POST"])
@require_client
def api_leave():
    body = request.get_json(silent=True, force=True) or {}
    fl_service.leave_session(g.client, str(body.get("reason", "left"))[:100] if isinstance(body, dict) else "left")
    return jsonify({"message": "Left session"}), 200


@app.route("/api/clients/status", methods=["POST"])
@require_client
def api_client_status():
    body = json_body()
    fl_service.set_client_state(g.client, body.get("state"))
    return jsonify({"message": "Status updated", "status": g.client.status}), 200


@app.route("/api/clients", methods=["GET"])
def api_clients():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"total_clients": 0, "clients": []}), 200
    ov = fl_service.session_overview(sess)
    return jsonify({"session_id": sess.session_id, "total_clients": len(ov["clients"]), "clients": ov["clients"]}), 200


@app.route("/api/sessions/current", methods=["GET"])
def api_session_current():
    """One call for the dashboard / status bar (poll every <= 3 s, NFR-P05)."""
    cid = request.args.get("client_id")
    sess = None
    if cid and not request.args.get("session_id"):
        c = Client.query.filter_by(client_id=cid).first()       # a client's own page: its own training
        sess = fl_service.get_session(c.session_id) if c and c.session_id else None
    return jsonify(fl_service.session_overview(sess or _session_for_metrics(), cid)), 200


@app.route("/api/sessions/joinable", methods=["GET"])
def api_sessions_joinable():
    """The training that is running right now (the training page shows 'join the running training')."""
    return jsonify({"sessions": fl_service.joinable_summary()}), 200


@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    rows = ExperimentSession.query.order_by(ExperimentSession.created_at.desc()).limit(20).all()
    return jsonify({"sessions": [s.to_dict() for s in rows]}), 200


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_session_by_id(session_id):
    sess = fl_service.get_session(session_id)
    if sess is None:
        raise not_found("Session not found", "session_not_found")
    return jsonify(fl_service.session_overview(sess)), 200


# ── rounds ───────────────────────────────────────────────────────────────

@app.route("/api/rounds/current", methods=["GET"])
@require_client
def api_round_current():
    return jsonify(fl_service.current_round_view(g.client)), 200


@app.route("/api/rounds/submit", methods=["POST"])
@app.route("/api/rounds/<int:round_number>/submit", methods=["POST"])
@require_client
def api_round_submit(round_number=None):
    data = json_body()
    if round_number is not None:
        data.setdefault("round", round_number)
    result = fl_service.submit_update(g.client, data, payload_bytes=request.content_length or 0)
    return jsonify(result), 200


@app.route("/api/rounds/history", methods=["GET"])
def api_round_history():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"rounds": []}), 200
    rows = Round.query.filter_by(session_id=sess.session_id).order_by(Round.id).all()
    return jsonify({"session_id": sess.session_id, "rounds": [r.to_dict() for r in rows]}), 200


# ── model & data ─────────────────────────────────────────────────────────

@app.route("/api/model/architecture", methods=["GET"])
def api_model_architecture():
    return jsonify(model_spec.architecture()), 200


@app.route("/api/model/global", methods=["GET"])
@require_client
def api_model_global():
    since = request.args.get("since")
    return jsonify(fl_service.global_model_payload(g.client.session_id, int(since) if since not in (None, "") else None)), 200


@app.route("/api/dataset/shard", methods=["GET"])
@require_client
def api_dataset_shard():
    return jsonify(fl_service.client_shard_payload(g.client)), 200


# ── metrics, selection, export ───────────────────────────────────────────

@app.route("/api/metrics/rounds", methods=["GET"])
def api_metrics_rounds():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"session_id": None, "rounds": []}), 200
    logs = ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number).all()
    return jsonify({"session_id": sess.session_id, "rounds": [l.to_dict() for l in logs]}), 200


@app.route("/api/metrics/latest", methods=["GET"])
def api_metrics_latest():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"latest": None}), 200
    log = ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number.desc()).first()
    return jsonify({"session_id": sess.session_id, "latest": log.to_dict() if log else None}), 200


@app.route("/api/metrics/clients", methods=["GET"])
def api_metrics_clients():
    return jsonify({"rows": _client_metric_rows(_session_for_metrics())}), 200


@app.route("/api/selection", methods=["GET"])
def api_selection():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"selection": []}), 200
    rows = ClientSelectionLog.query.filter_by(session_id=sess.session_id).order_by(ClientSelectionLog.id).all()
    return jsonify({"session_id": sess.session_id, "selection": [r.to_dict() for r in rows]}), 200


@app.route("/api/events", methods=["GET"])
def api_events():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"events": []}), 200
    rows = ClientEvent.query.filter_by(session_id=sess.session_id).order_by(ClientEvent.id.desc()).limit(100).all()
    return jsonify({"events": [r.to_dict() for r in rows]}), 200


def _client_metric_rows(sess):
    rows = []
    if not sess:
        return rows
    logs = ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number).all()
    for l in logs:
        for m in json.loads(l.client_metrics_json or "[]"):
            rows.append({"session_id": sess.session_id, "round": l.round_number, "timestamp": iso(l.logged_at), **m})
    return rows


_CSV_COLUMNS = [
    "session_id", "round", "timestamp", "accuracy", "loss", "precision", "recall", "f1",
    "avg_local_accuracy", "avg_local_loss", "client_count", "participating_clients",
    "payload_bytes", "aggregation_ms", "round_duration_seconds", "aggregation_method",
]


def _csv_response(rows, columns, filename):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(columns)
    for r in rows:
        w.writerow([r.get(c, "") if r.get(c) is not None else "" for c in columns])
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _rounds_for_export(sess):
    if not sess:
        return []
    logs = ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number).all()
    out = []
    for l in logs:
        d = l.to_dict()
        d["participating_clients"] = ";".join(d["participating_clients"])
        out.append(d)
    return out


@app.route("/api/export/csv", methods=["GET"])
@app.route("/experiment/export/csv", methods=["GET"])
def api_export_csv():
    sess = _session_for_metrics()
    return _csv_response(_rounds_for_export(sess), _CSV_COLUMNS, "synora_experiment_results.csv")


@app.route("/api/export/clients.csv", methods=["GET"])
def api_export_clients_csv():
    cols = ["session_id", "round", "timestamp", "client_name", "language", "backend", "dataset_size",
            "local_accuracy", "local_loss", "train_seconds", "payload_bytes"]
    return _csv_response(_client_metric_rows(_session_for_metrics()), cols, "synora_client_metrics.csv")


@app.route("/api/export/json", methods=["GET"])
@app.route("/experiment/export/json", methods=["GET"])
def api_export_json():
    sess = _session_for_metrics()
    payload = {
        "export_time": iso(utcnow()),
        "session": sess.to_dict() if sess else None,
        "model": fl_service._model_info(),
        "rounds": [l.to_dict() for l in ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number)] if sess else [],
        "selection_log": [r.to_dict() for r in ClientSelectionLog.query.filter_by(session_id=sess.session_id).order_by(ClientSelectionLog.id)] if sess else [],
    }
    return Response(
        json.dumps(payload, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": 'attachment; filename="synora_experiment_results.json"'},
    )


# ═════════════════════════════════════════════════════════════════════════
# Legacy routes (kept so older scripts / Postman collections keep working)
# ═════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health_check():
    try:
        count = Client.query.count()
        return jsonify({"status": "healthy", "database": "connected", "registered_clients": count, "timestamp": iso(utcnow())}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/register", methods=["POST"])
def register_client():
    """Legacy registration: joins the running session (language partitioning)."""
    data = json_body()
    if "client_name" not in data:
        raise bad_request("client_name is required", "client_name")
    pref = data_service.PARTITION_TO_LANGUAGE.get(data.get("partition"))
    result = fl_service.join_session(
        {"partition_type": "language"},
        {"client_name": data["client_name"], "preferred_language": pref},
    )
    c = result["client"]
    return jsonify({"message": "Registration successful", "client_id": c.client_id, "partition": c.partition,
                    "status": c.status, "session_token": c.session_token, "session_id": c.session_id}), 200


@app.route("/clients", methods=["GET"])
def get_clients():
    sess = _session_for_metrics()
    clients = Client.query.filter_by(session_id=sess.session_id).all() if sess else []
    return jsonify({"total_clients": len(clients), "clients": {c.client_id: c.to_dict() for c in clients}}), 200


@app.route("/clients/reset", methods=["DELETE"])
@require_admin
def reset_clients():
    Client.query.delete()
    db.session.commit()
    return jsonify({"message": "All clients cleared"}), 200


@app.route("/global-model", methods=["GET"])
@app.route("/model/global", methods=["GET"])
def get_global_model_legacy():
    """Legacy shape: weights as [{shape, data:[...]}] (fixes the old format mismatch)."""
    sess = fl_service.latest_session()
    if sess is None:
        w = model_spec.init_weights(DEFAULT_SEED)
        return jsonify({"version": 0, "round": 0, "architecture": model_spec.architecture(),
                        "weights": weight_codec.encode_weights_legacy(w)}), 200
    payload = fl_service.global_model_payload(sess.session_id)
    ck_w = weight_codec.decode_weights(payload["weights"])
    payload["weights"] = weight_codec.encode_weights_legacy(ck_w)
    return jsonify(payload), 200


DEFAULT_SEED = fl_service.DEFAULT_CONFIG["seed"]


@app.route("/model/version", methods=["GET"])
def get_model_version():
    sess = fl_service.latest_session()
    return jsonify({"version": sess.model_version if sess else 0}), 200


@app.route("/global-model/ready", methods=["GET"])
def check_model_ready():
    sess = fl_service.latest_session()
    return jsonify({"ready": True, "version": sess.model_version if sess else 0,
                    "round": sess.current_round if sess else 0, "has_weights": True}), 200


@app.route("/submit-update", methods=["POST"])
@app.route("/model/update", methods=["POST"])
def legacy_submit():
    data = json_body()
    cid = data.pop("client_id", None) or data.pop("clientId", None)
    client = Client.query.filter_by(client_id=cid).first() if cid else None
    if client is None or client.status == "left":
        raise unauthorized("Unknown client_id - register first")
    return jsonify(fl_service.submit_update(client, data, payload_bytes=request.content_length or 0)), 200


@app.route("/config/threshold", methods=["POST"])
def set_threshold():
    data = json_body()
    if "min_clients" not in data:
        raise bad_request("min_clients is required", "min_clients")
    try:
        n = int(data["min_clients"])
        if n < 1:
            raise ValueError
    except (TypeError, ValueError):
        raise bad_request("min_clients must be an integer >= 1", "min_clients")
    cfg = AggregationConfig.query.first() or AggregationConfig(min_clients=n, max_rounds=20, current_round=0)
    cfg.min_clients = n
    db.session.add(cfg)
    db.session.commit()
    return jsonify({"message": f"Threshold set to {n}", "config": cfg.to_dict(),
                    "note": "The threshold of a RUNNING session is fixed when the session starts."}), 200


@app.route("/config/threshold", methods=["GET"])
def get_threshold():
    sess = fl_service.latest_session()
    if sess:
        c = sess.config()
        return jsonify({"min_clients": c["min_participation_threshold"], "max_rounds": c["total_rounds"],
                        "current_round": sess.current_round}), 200
    cfg = AggregationConfig.query.first()
    return jsonify(cfg.to_dict() if cfg else {"min_clients": 1, "max_rounds": 20, "current_round": 0}), 200


@app.route("/aggregate/check", methods=["GET"])
def check_aggregation():
    """Read-only status (the old version incremented the round on every GET)."""
    sess = fl_service.latest_session()
    if not sess:
        return jsonify({"can_aggregate": False, "connected_clients": 0, "min_required": 1, "current_round": 0}), 200
    c = sess.config()
    live = len(fl_service.live_clients(sess.session_id))
    return jsonify({"can_aggregate": live >= c["min_participation_threshold"], "connected_clients": live,
                    "min_required": c["min_participation_threshold"], "current_round": sess.current_round}), 200


@app.route("/config/selection", methods=["POST"])
def set_selection_count():
    data = json_body()
    try:
        count = int(data["selection_count"])
        if count < 1:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise bad_request("selection_count must be an integer >= 1", "selection_count")
    app.config["SELECTION_COUNT"] = count
    return jsonify({"message": f"Selection count set to {count}", "selection_count": count}), 200


@app.route("/select/clients", methods=["GET"])
def select_clients():
    sess = fl_service.latest_session()
    if not sess:
        raise bad_request("No clients registered", code="no_clients")
    rnd = fl_service.latest_round(sess.session_id, sess.current_round) if sess.current_round else None
    ids = rnd.selected_list() if rnd else []
    clients = Client.query.filter(Client.client_id.in_(ids)).all() if ids else []
    return jsonify({"round": sess.current_round, "selected_count": len(clients),
                    "selected_clients": {c.client_id: c.to_dict() for c in clients},
                    "total_available": len(fl_service.live_clients(sess.session_id))}), 200


@app.route("/select/history", methods=["GET"])
def get_selection_history():
    sess = _session_for_metrics()
    if not sess:
        return jsonify({"total_rounds": 0, "participation_counts": {}, "participation_rates": {}}), 200
    rows = ClientSelectionLog.query.filter_by(session_id=sess.session_id, selected=True).all()
    counts = {}
    for r in rows:
        counts[r.client_name] = counts.get(r.client_name, 0) + 1
    total = ExperimentLog.query.filter_by(session_id=sess.session_id).count()
    return jsonify({"total_rounds": total, "participation_counts": counts,
                    "participation_rates": {k: (round(v / total, 4) if total else 0) for k, v in counts.items()}}), 200


@app.route("/experiment/config", methods=["POST"])
def set_experiment_config():
    data = json_body()
    cfg = fl_service.normalize_config(data)
    rec = ExperimentConfig(
        experiment_name=data.get("experiment_name", "kenyan_fl_experiment"),
        num_rounds=cfg["total_rounds"], num_clients=cfg["max_clients_per_round"],
        min_clients_per_round=cfg["min_participation_threshold"], learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"], local_epochs=cfg["local_epochs"],
        partition_type=cfg["partition_type"], dirichlet_alpha=cfg["dirichlet_alpha"],
        languages=",".join(data.get("languages", ["dholuo", "kalenjin", "kidawida"])),
    )
    db.session.add(rec)
    db.session.commit()
    return jsonify({"message": "Experiment config saved", "config": rec.to_dict()}), 200


@app.route("/experiment/config", methods=["GET"])
def get_experiment_config():
    sess = fl_service.latest_session()
    if sess:
        return jsonify(sess.config()), 200
    rec = ExperimentConfig.query.order_by(ExperimentConfig.created_at.desc()).first()
    if not rec:
        raise not_found("No experiment config saved yet")
    return jsonify(rec.to_dict()), 200


@app.route("/experiment/summary", methods=["GET"])
def get_summary():
    sess = _session_for_metrics()
    rounds = [l.to_dict() for l in ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number)] if sess else []
    return jsonify({"experiment_name": "kenyan_fl_experiment", "total_rounds": len(rounds),
                    "configuration": sess.config() if sess else {}, "rounds": rounds,
                    "start_time": rounds[0]["timestamp"] if rounds else None, "export_time": iso(utcnow())}), 200


@app.route("/experiment/reset", methods=["DELETE"])
@require_admin
def reset_experiment():
    ExperimentLog.query.delete()
    db.session.commit()
    return jsonify({"message": "Experiment logs cleared"}), 200


@app.route("/dataset/partition/<partition>", methods=["GET"])
def get_dataset_partition(partition):
    """Legacy: raw sentences for one language (the v2 client uses /api/dataset/shard)."""
    language = data_service.PARTITION_TO_LANGUAGE.get(partition)
    if not language:
        raise not_found(f"No language mapped for '{partition}'", "unknown_partition")
    from preprocessing.loader import load_dataset

    rows = load_dataset(language.lower())[: model_spec.SAMPLES_PER_CLIENT]
    return jsonify({"partition": partition, "language": language, "texts": [r[0] for r in rows],
                    "labels": [r[1] for r in rows], "count": len(rows)}), 200


# ── entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("=" * 55)
    print("  Synora Coordination Server")
    print(f"  Running at: http://127.0.0.1:{port}")
    print("=" * 55)
    data_service.get_data()  # warm the dataset cache
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
