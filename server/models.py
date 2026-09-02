"""
Synora — Database Models
========================
SQLAlchemy models matching the Supabase tables.
Each model maps directly to one table.
"""

from datetime import datetime
from server.database import db


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

    def to_dict(self):
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "partition": self.partition,
            "status": self.status,
            "registered_at": self.registered_at.isoformat(),
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

    def to_dict(self):
        return {
            "round_number": self.round_number,
            "state": self.state,
            "min_clients": self.min_clients,
            "start_time": self.start_time.isoformat(),
            "end_time": (self.end_time.isoformat() if self.end_time else None),
            "duration_seconds": self.duration_seconds,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss,
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
            "created_at": self.created_at.isoformat(),
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

    def to_dict(self):
        return {
            "round": self.round_number,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "participating_clients": (self.participating_clients.split(",")),
            "client_count": self.client_count,
            "timestamp": self.logged_at.isoformat(),
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
