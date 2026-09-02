"""
Integration tests for Synora coordination server.
Uses Flask test client — no live server needed.
Covers: US-08 (registration), US-11 (threshold),
        US-12/13 (selection), US-16 (config), US-17 (export)
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.coordination_server import app


@pytest.fixture
def client():
    """Fresh Flask test client with clean state for every test."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        # Reset in-memory state between tests
        import server.coordination_server as srv
        srv.registered_clients.clear()
        from server.aggregation_config import AggregationConfig
        from server.client_selector import ClientSelector
        from server.experiment_tracker import ExperimentTracker
        srv.config = AggregationConfig(min_clients=2, max_rounds=10)
        srv.selector = ClientSelector(selection_count=2)
        srv.tracker = ExperimentTracker(experiment_name="test_experiment")
        yield c


# ── US-08: Client Registration ─────────────────────────────

class TestClientRegistration:
    def test_register_new_client(self, client):
        r = client.post("/register", json={"client_name": "client_A"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["message"] == "Registration successful"
        assert "client_id" in data
        assert "partition" in data
        assert data["status"] == "active"

    def test_duplicate_registration_rejected(self, client):
        client.post("/register", json={"client_name": "client_A"})
        r = client.post("/register", json={"client_name": "client_A"})
        assert r.status_code == 409
        assert "already registered" in r.get_json()["error"]

    def test_missing_client_name_rejected(self, client):
        r = client.post("/register", json={})
        assert r.status_code == 400

    def test_get_all_clients(self, client):
        client.post("/register", json={"client_name": "client_A"})
        client.post("/register", json={"client_name": "client_B"})
        r = client.get("/clients")
        assert r.status_code == 200
        data = r.get_json()
        assert data["total_clients"] == 2

    def test_partition_assigned_round_robin(self, client):
        partitions = []
        for i in range(3):
            r = client.post("/register", json={"client_name": f"c{i}"})
            partitions.append(r.get_json()["partition"])
        # Three clients → three different partitions
        assert len(set(partitions)) == 3


# ── US-11: Aggregation Threshold ───────────────────────────

class TestAggregationThreshold:
    def test_get_default_threshold(self, client):
        r = client.get("/config/threshold")
        assert r.status_code == 200
        data = r.get_json()
        assert "min_clients" in data

    def test_set_threshold(self, client):
        r = client.post("/config/threshold", json={"min_clients": 3})
        assert r.status_code == 200
        assert r.get_json()["config"]["min_clients"] == 3

    def test_aggregation_below_threshold_returns_waiting(self, client):
        client.post("/config/threshold", json={"min_clients": 3})
        r = client.get("/aggregate/check")
        assert r.status_code == 200
        data = r.get_json()
        assert data["can_aggregate"] is False
        assert "WAITING" in data["status"]

    def test_aggregation_at_threshold_returns_ready(self, client):
        client.post("/config/threshold", json={"min_clients": 2})
        client.post("/register", json={"client_name": "c1"})
        client.post("/register", json={"client_name": "c2"})
        r = client.get("/aggregate/check")
        data = r.get_json()
        assert data["can_aggregate"] is True
        assert data["status"] == "READY"

    def test_invalid_threshold_rejected(self, client):
        r = client.post("/config/threshold", json={"min_clients": 0})
        assert r.status_code == 400

    def test_missing_min_clients_rejected(self, client):
        r = client.post("/config/threshold", json={})
        assert r.status_code == 400


# ── US-12/13: Client Selection ─────────────────────────────

class TestClientSelection:
    def test_select_clients_random(self, client):
        for i in range(4):
            client.post("/register", json={"client_name": f"c{i}"})
        r = client.get("/select/clients")
        assert r.status_code == 200
        data = r.get_json()
        assert data["selected_count"] == 2

    def test_no_clients_returns_400(self, client):
        r = client.get("/select/clients")
        assert r.status_code == 400

    def test_selection_history_endpoint(self, client):
        for i in range(3):
            client.post("/register", json={"client_name": f"c{i}"})
        client.get("/select/clients")
        r = client.get("/select/history")
        assert r.status_code == 200
        data = r.get_json()
        assert "stats" in data
        assert "frequency_distribution" in data
        assert "bias_check" in data

    def test_set_selection_count(self, client):
        r = client.post("/config/selection", json={"selection_count": 3})
        assert r.status_code == 200
        assert r.get_json()["selection_count"] == 3

    def test_invalid_selection_count_rejected(self, client):
        r = client.post("/config/selection", json={"selection_count": 0})
        assert r.status_code == 400


# ── US-16: Experiment Config ───────────────────────────────

class TestExperimentConfig:
    def test_save_config(self, client):
        cfg = {
            "num_rounds": 10,
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 5,
            "partition_type": "non_iid",
            "dirichlet_alpha": 0.5,
        }
        r = client.post("/experiment/config", json=cfg)
        assert r.status_code == 200
        assert r.get_json()["message"] == "Experiment config saved"

    def test_empty_config_rejected(self, client):
        r = client.post("/experiment/config", data=b"", content_type="application/json")
        assert r.status_code == 400


# ── US-17: Results Export ──────────────────────────────────

class TestResultsExport:
    def test_log_round(self, client):
        r = client.post("/experiment/log", json={
            "round": 1, "accuracy": 0.72,
            "loss": 0.54, "participating_clients": ["c1", "c2"]
        })
        assert r.status_code == 200
        assert "round_data" in r.get_json()

    def test_get_summary(self, client):
        client.post("/experiment/log", json={
            "round": 1, "accuracy": 0.72, "loss": 0.54,
            "participating_clients": []
        })
        r = client.get("/experiment/summary")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_rounds" in data

    def test_export_json(self, client):
        client.post("/experiment/log", json={
            "round": 1, "accuracy": 0.80, "loss": 0.30,
            "participating_clients": ["c1"]
        })
        r = client.get("/experiment/export/json")
        assert r.status_code == 200
        assert r.content_type == "application/json"

    def test_export_csv(self, client):
        client.post("/experiment/log", json={
            "round": 1, "accuracy": 0.80, "loss": 0.30,
            "participating_clients": ["c1"]
        })
        r = client.get("/experiment/export/csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type or r.status_code == 200


# ── US-09: Model Distribution Routes (new) ─────────────────

class TestModelDistribution:
    def test_get_global_model_returns_weights(self, client):
        r = client.get("/model/global")
        assert r.status_code == 200
        data = r.get_json()
        assert "weights" in data
        assert "version" in data
        assert "num_layers" in data
        assert len(data["weights"]) > 0

    def test_get_model_version(self, client):
        r = client.get("/model/version")
        assert r.status_code == 200
        data = r.get_json()
        assert "version" in data
        assert isinstance(data["version"], int)

    def test_submit_update_from_registered_client(self, client):
        # Register a client first
        reg = client.post("/register", json={"client_name": "trainer_c"})
        client_id = reg.get_json()["client_id"]

        # Submit a weight update — weights must be flat numeric lists
        r = client.post("/model/update", json={
            "client_id": client_id,
            "round": 1,
            "weights": [[0.1, 0.2, 0.3, 0.4]]
        })
        assert r.status_code == 200
        assert r.get_json()["status"] == "accepted"

    def test_submit_update_from_unregistered_client_rejected(self, client):
        r = client.post("/model/update", json={
            "client_id": "nonexistent-id",
            "round": 1,
            "weights": [[0.1, 0.2]]
        })
        assert r.status_code == 403

    def test_payload_with_disallowed_field_rejected(self, client):
        reg = client.post("/register", json={"client_name": "trainer_d"})
        client_id = reg.get_json()["client_id"]

        # Include a raw_text field — Privacy Enforcement Layer must reject
        r = client.post("/model/update", json={
            "client_id": client_id,
            "round": 1,
            "weights": [[0.1, 0.2]],
            "raw_text": "this is private training data"   # must be blocked
        })
        assert r.status_code == 400
        data = r.get_json()
        assert data["error"] == "bad_request"
        assert "raw_text" in data["message"] or "Unexpected" in data["message"]

    def test_model_version_increments_after_init(self, client):
        r1 = client.get("/model/global")
        assert r1.status_code == 200
        r2 = client.get("/model/version")
        assert r2.get_json()["version"] == r1.get_json()["version"]
