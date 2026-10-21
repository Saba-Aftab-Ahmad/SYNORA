"""
Synora — Database Models
========================
SQLAlchemy models matching the Supabase tables.
Each model maps directly to one table.
"""

from datetime import datetime, timezone
from server.database import db
from server.training_settings import DEFAULT_CONFIG


def iso(dt):
    """
    ISO-8601 UTC timestamp WITH millisecond precision and a trailing 'Z'.

    SRS NFR-R02 requires millisecond-accurate timestamps.  The old code used
    naive `datetime.utcnow().isoformat()` (no 'Z'), which browsers parse as
    LOCAL time -> wrong clock in the experiment log for users outside UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def utcnow():
    """Naive UTC 'now' (all DB DateTime columns store naive UTC)."""
    return datetime.utcnow()


class Client(db.Model):
    """
    Registered browser client.
    Maps to: clients table
    User Story: US-08
    """

    __tablename__ = "clients"

    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(36), unique=True, nullable=False)
    client_name = db.Column(db.String(100), unique=True, nullable=False)
    partition = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default="active")
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ── v2 columns (multi-client sessions, capability reporting, auth) ──
    # All nullable so the existing Supabase table can be migrated in place.
    session_id = db.Column(db.String(36), nullable=True, index=True)
    partition_index = db.Column(db.Integer, nullable=True)
    language = db.Column(db.String(20), nullable=True)
    session_token = db.Column(db.String(64), nullable=True)
    cpu_cores = db.Column(db.Integer, nullable=True)
    memory_gb = db.Column(db.Float, nullable=True)
    backend = db.Column(db.String(20), nullable=True)          # webgpu | webgl | cpu
    webgpu_available = db.Column(db.Boolean, nullable=True)
    webnn_available = db.Column(db.Boolean, nullable=True)
    capability_score = db.Column(db.Float, nullable=True)
    user_agent = db.Column(db.String(200), nullable=True)
    last_seen = db.Column(db.DateTime, nullable=True)
    # True once the browser has loaded the model + its data and started polling.
    # Only ready clients are put into a round, so a newcomer that is still
    # loading never makes the clients that are already training wait.
    ready = db.Column(db.Boolean, nullable=True)
    # How many rounds THIS client wants to take part in (its own choice) and the
    # global round that was running when it joined (0 = it started the training).
    rounds_wanted = db.Column(db.Integer, nullable=True)
    joined_round = db.Column(db.Integer, nullable=True)
    # This client's OWN data partitioning (its choice; independent of other clients).
    partition_type = db.Column(db.String(20), nullable=True)
    dirichlet_alpha = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "partition": self.partition,
            "status": self.status,
            "registered_at": iso(self.registered_at),
            "session_id": self.session_id,
            "partition_index": self.partition_index,
            "language": self.language,
            "backend": self.backend,
            "capability_score": self.capability_score,
            "last_seen": iso(self.last_seen),
            "rounds_wanted": self.rounds_wanted,
            "joined_round": self.joined_round,
            "partition_type": self.partition_type,
            "dirichlet_alpha": self.dirichlet_alpha,
        }


class Round(db.Model):
    """
    Federated training round.
    Maps to: rounds table
    User Story: US-07, US-11
    """

    __tablename__ = "rounds"

    id = db.Column(db.Integer, primary_key=True)
    round_number = db.Column(db.Integer, nullable=False)
    state = db.Column(db.String(20), default="initialised")
    min_clients = db.Column(db.Integer, default=2)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    global_accuracy = db.Column(db.Float, nullable=True)
    global_loss = db.Column(db.Float, nullable=True)

    # ── v2 columns ──
    session_id = db.Column(db.String(36), nullable=True, index=True)
    selected_clients = db.Column(db.Text, nullable=True)        # comma separated client_ids
    deadline = db.Column(db.DateTime, nullable=True)
    attempt = db.Column(db.Integer, nullable=True)
    aggregating_since = db.Column(db.DateTime, nullable=True)

    def selected_list(self):
        return [c for c in (self.selected_clients or "").split(",") if c]

    def to_dict(self):
        return {
            "round_number": self.round_number,
            "state": self.state,
            "min_clients": self.min_clients,
            "start_time": iso(self.start_time),
            "end_time": iso(self.end_time),
            "deadline": iso(self.deadline),
            "duration_seconds": self.duration_seconds,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss,
            "session_id": self.session_id,
            "selected_clients": self.selected_list(),
        }


class ExperimentConfig(db.Model):
    """
    FL experiment hyperparameter configuration.
    Maps to: experiment_config table
    User Story: US-16
    """

    __tablename__ = "experiment_config"

    id = db.Column(db.Integer, primary_key=True)
    experiment_name = db.Column(db.String(100), default="kenyan_fl_experiment")
    num_rounds = db.Column(db.Integer, default=10)
    num_clients = db.Column(db.Integer, default=3)
    min_clients_per_round = db.Column(db.Integer, default=2)
    learning_rate = db.Column(db.Float, default=0.01)
    batch_size = db.Column(db.Integer, default=32)
    local_epochs = db.Column(db.Integer, default=5)
    partition_type = db.Column(db.String(20), default="non_iid")
    dirichlet_alpha = db.Column(db.Float, default=0.5)
    languages = db.Column(db.Text, default="dholuo,kalenjin,kidawida")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "experiment_name": self.experiment_name,
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "min_clients_per_round": self.min_clients_per_round,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "partition_type": self.partition_type,
            "dirichlet_alpha": self.dirichlet_alpha,
            "languages": self.languages.split(","),
            "created_at": iso(self.created_at),
        }


class ExperimentLog(db.Model):
    """
    Per-round training metrics log.
    Maps to: experiment_logs table
    User Story: US-17
    """

    __tablename__ = "experiment_logs"

    id = db.Column(db.Integer, primary_key=True)
    round_number = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    loss = db.Column(db.Float, nullable=False)
    participating_clients = db.Column(db.Text, nullable=False)
    client_count = db.Column(db.Integer, nullable=False)
    logged_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ── v2 columns: real evaluation + per-round telemetry ──
    session_id = db.Column(db.String(36), nullable=True, index=True)
    precision_macro = db.Column(db.Float, nullable=True)
    recall_macro = db.Column(db.Float, nullable=True)
    f1_macro = db.Column(db.Float, nullable=True)
    avg_local_accuracy = db.Column(db.Float, nullable=True)
    avg_local_loss = db.Column(db.Float, nullable=True)
    payload_bytes = db.Column(db.Integer, nullable=True)
    aggregation_ms = db.Column(db.Float, nullable=True)
    round_duration_seconds = db.Column(db.Float, nullable=True)
    aggregation_method = db.Column(db.String(30), nullable=True)
    per_class_json = db.Column(db.Text, nullable=True)
    client_metrics_json = db.Column(db.Text, nullable=True)

    def to_dict(self):
        import json as _json

        def _load(v):
            try:
                return _json.loads(v) if v else None
            except Exception:
                return None

        return {
            "round": self.round_number,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "precision": self.precision_macro,
            "recall": self.recall_macro,
            "f1": self.f1_macro,
            "avg_local_accuracy": self.avg_local_accuracy,
            "avg_local_loss": self.avg_local_loss,
            "payload_bytes": self.payload_bytes,
            "aggregation_ms": self.aggregation_ms,
            "round_duration_seconds": self.round_duration_seconds,
            "aggregation_method": self.aggregation_method,
            "participating_clients": [c for c in (self.participating_clients or "").split(",") if c],
            "client_count": self.client_count,
            "per_class": _load(self.per_class_json),
            "client_metrics": _load(self.client_metrics_json),
            "session_id": self.session_id,
            "timestamp": iso(self.logged_at),
        }


class AggregationConfig(db.Model):
    """
    Aggregation threshold configuration.
    Maps to: aggregation_config table
    User Story: US-11
    """

    __tablename__ = "aggregation_config"

    id = db.Column(db.Integer, primary_key=True)
    min_clients = db.Column(db.Integer, default=2)
    max_rounds = db.Column(db.Integer, default=10)
    current_round = db.Column(db.Integer, default=0)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self):
        return {
            "min_clients": self.min_clients,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
        }


# ═════════════════════════════════════════════════════════════════════════
#  v2 tables — all coordination state lives in PostgreSQL (SRS NFR-R01,
#  NFR-R05, NFR-R07: the server must recover from a restart and must work
#  with several gunicorn workers).  Nothing is kept in process memory.
# ═════════════════════════════════════════════════════════════════════════

class ExperimentSession(db.Model):
    """One federated experiment (one run of N rounds with M clients)."""

    __tablename__ = "experiment_sessions"

    session_id = db.Column(db.String(36), primary_key=True)
    status = db.Column(db.String(20), default="pending")       # pending|active|completed|aborted
    config_json = db.Column(db.Text, nullable=False)
    current_round = db.Column(db.Integer, default=0)
    model_version = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    ended_at = db.Column(db.DateTime, nullable=True)

    def config(self):
        """Session settings; anything missing (older sessions) falls back to the server defaults."""
        import json as _json
        return {**DEFAULT_CONFIG, **_json.loads(self.config_json)}

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "status": self.status,
            "config": self.config(),
            "current_round": self.current_round,
            "model_version": self.model_version,
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "ended_at": iso(self.ended_at),
        }


class ClientUpdate(db.Model):
    """A weight update submitted by one client for one round."""

    __tablename__ = "client_updates"
    __table_args__ = (
        db.UniqueConstraint("session_id", "round_number", "client_id", name="uq_update_session_round_client"),
    )

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    round_number = db.Column(db.Integer, nullable=False)
    client_id = db.Column(db.String(36), nullable=False)
    dataset_size = db.Column(db.Integer, nullable=False)
    weights_blob = db.Column(db.Text, nullable=True)           # cleared once aggregated
    local_accuracy = db.Column(db.Float, nullable=True)
    local_loss = db.Column(db.Float, nullable=True)
    train_seconds = db.Column(db.Float, nullable=True)
    payload_bytes = db.Column(db.Integer, nullable=True)
    backend = db.Column(db.String(20), nullable=True)
    local_epochs = db.Column(db.Integer, nullable=True)
    # the client's OWN model of this round, evaluated by the server on held-out data
    val_accuracy = db.Column(db.Float, nullable=True)
    per_class_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ModelCheckpoint(db.Model):
    """Global model after each round (round 0 = server-initialised weights)."""

    __tablename__ = "model_checkpoints"
    __table_args__ = (
        db.UniqueConstraint("session_id", "round_number", name="uq_checkpoint_session_round"),
    )

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    round_number = db.Column(db.Integer, nullable=False)
    weights_blob = db.Column(db.Text, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)
    num_clients = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ClientSelectionLog(db.Model):
    """Which clients were selected (or skipped) in every round, and why (US-12/13)."""

    __tablename__ = "client_selection_logs"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    round_number = db.Column(db.Integer, nullable=False)
    client_id = db.Column(db.String(36), nullable=False)
    client_name = db.Column(db.String(100), nullable=True)
    strategy = db.Column(db.String(20), nullable=True)
    selected = db.Column(db.Boolean, default=False)
    capability_score = db.Column(db.Float, nullable=True)
    reason = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "round": self.round_number,
            "client_id": self.client_id,
            "client_name": self.client_name,
            "strategy": self.strategy,
            "selected": bool(self.selected),
            "capability_score": self.capability_score,
            "reason": self.reason,
            "timestamp": iso(self.created_at),
        }


class ClientEvent(db.Model):
    """Audit trail: join / leave / pause / timeout / dropped / rejected update."""

    __tablename__ = "client_events"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=True, index=True)
    client_id = db.Column(db.String(36), nullable=True)
    event_type = db.Column(db.String(30), nullable=False)
    detail = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "client_id": self.client_id,
            "event_type": self.event_type,
            "detail": self.detail,
            "timestamp": iso(self.created_at),
        }
