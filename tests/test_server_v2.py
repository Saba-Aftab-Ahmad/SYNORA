"""
In-process integration tests for the coordination server (v3: join-anytime).
No live server and no Supabase needed.

Covers: US-06/08 join + capabilities, US-07/10 rounds + FedAvg, US-11 threshold
(fixed to 1), US-12/13 automatic selection, US-16 config validation (rounds +
data split only), US-17 export, late joiners, pause/resume, multi-client
concurrency, timeout/exclusion, recovery, auth (401), privacy layer (400),
SDS error schema, ms timestamps (NFR-R02), global (combined) summary.
"""
import os
import re
import sys
import threading
from datetime import timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import fl_service, model_spec, weight_codec
from server.coordination_server import app
from server.database import db
from server.models import Client, ClientUpdate, ExperimentSession, Round, utcnow


@pytest.fixture(autouse=True)
def clean_db():
    with app.app_context():
        for t in reversed(db.metadata.sorted_tables):
            db.session.execute(t.delete())
        db.session.commit()
    yield


def api():
    return app.test_client()


def join(c, config=None, caps=None, **extra):
    body = {"config": config or {}, "capabilities": caps or {"backend": "cpu", "cpu_cores": 4, "memory_gb": 8}, **extra}
    r = c.post("/api/clients/join", json=body)
    assert r.status_code == 200, r.json
    return r.json["client"]["session_token"], r.json


def H(tok):
    return {"X-Session-Token": tok}


FAST = {"total_rounds": 3}


def warm(c, *toks):
    """A client becomes 'ready' (eligible for rounds) after its first poll, like the browser."""
    for t in toks:
        poll(c, t)


def cfg_with(**kw):
    """Change the stored session settings (e.g. the per-round cap) of the only running session."""
    with app.app_context():
        sess = ExperimentSession.query.order_by(ExperimentSession.created_at.desc()).first()
        import json as _j
        c = _j.loads(sess.config_json); c.update(kw); sess.config_json = _j.dumps(c)
        db.session.commit()


def weights_for(c, tok):
    return c.get("/api/model/global", headers=H(tok)).json["weights"]


def submit(c, tok, rn, w, acc=0.5):
    return c.post("/api/rounds/submit", headers=H(tok), json={"round": rn, "weights": w, "metrics": {"accuracy": acc, "loss": 1.0}})


def poll(c, tok):
    return c.get("/api/rounds/current", headers=H(tok)).json


# ── model / spec ─────────────────────────────────────────────────────────

def test_model_is_lightweight_and_within_5mb():
    a = api().get("/api/model/architecture").json
    assert a["total_parameters"] == 160579
    assert a["model_size_bytes"] < 5 * 1024 * 1024          # NFR-A03
    assert [l["layer_name"] for l in a["layers"]][0] == "embedding/embeddings"


def test_server_initialises_global_model_round0():
    c = api()
    tok, _ = join(c, FAST)
    m = c.get("/api/model/global", headers=H(tok)).json
    assert m["version"] == 0 and len(m["weights"]) == 5
    arrs = weight_codec.decode_weights(m["weights"])
    ref = model_spec.init_weights(42)
    assert np.allclose(arrs[1], ref[1])                      # deterministic server init


# ── auth / errors / privacy ──────────────────────────────────────────────

def test_missing_and_bad_token_return_401_with_sds_schema():
    c = api()
    for headers in ({}, {"X-Session-Token": "nope"}):
        r = c.get("/api/rounds/current", headers=headers)
        assert r.status_code == 401
        assert set(r.json) >= {"error", "message", "field", "reference_id"}


def test_unknown_route_returns_json_404():
    r = api().get("/api/nothing")
    assert r.status_code == 404 and r.json["error"] == "not_found"


def test_invalid_config_rejected_400():
    c = api()
    r = c.post("/api/clients/join", json={"config": {"total_rounds": 0}})
    assert r.status_code == 400 and r.json["field"] == "total_rounds"
    r = c.post("/api/clients/join", json={"config": {"aggregation_method": "fedprox"}})
    assert r.status_code == 400
    r = c.post("/api/clients/join", json={"config": {"partition_type": "weird"}})
    assert r.status_code == 400 and r.json["field"] == "partition_type"


def test_privacy_layer_rejects_raw_text_and_bad_weights():
    c = api()
    tok, _ = join(c, FAST)
    w = weights_for(c, tok)
    poll(c, tok)
    r = c.post("/api/rounds/submit", headers=H(tok), json={"round": 1, "weights": w, "texts": ["hello"]})
    assert r.status_code == 400 and r.json["error"] == "privacy_violation"
    bad = [dict(t) for t in w]
    bad[0] = {**bad[0], "shape": [10, 10]}
    assert submit(c, tok, 1, bad).status_code == 400
    nan = np.full((5000, 32), np.nan, dtype="<f4")
    bad[0] = {**w[0], "data_b64": weight_codec.encode_array(nan)}
    r = submit(c, tok, 1, bad)
    assert r.status_code == 400 and "NaN" in r.json["message"]
    assert submit(c, tok, 1, w[:2]).status_code == 400
    text_w = [dict(t) for t in w]
    text_w[2] = {"shape": [16], "data": ["a"] * 16}
    assert submit(c, tok, 1, text_w).status_code == 400
    assert submit(c, tok, 1, w).status_code == 200            # a clean update still works


# ── rounds / FedAvg / evaluation ─────────────────────────────────────────

def test_single_client_full_session_and_real_metrics():
    c = api()
    tok, _ = join(c, FAST)
    for rn in (1, 2, 3):
        v = poll(c, tok)
        assert v["phase"] == "train" and v["round_number"] == rn
        r = submit(c, tok, rn, weights_for(c, tok))
        assert r.status_code == 200 and r.json["aggregated"] is True
    v = poll(c, tok)
    assert v["phase"] == "session_complete" and v["session_status"] == "completed"
    rounds = c.get("/api/metrics/rounds").json["rounds"]
    assert [r["round"] for r in rounds] == [1, 2, 3]
    r0 = rounds[0]
    assert 0 <= r0["accuracy"] <= 1 and r0["f1"] is not None and len(r0["per_class"]) == 3
    assert r0["client_metrics"][0]["local_accuracy"] == 0.5


def test_one_client_is_enough_nothing_is_waited_for():
    """The training starts the moment the first client joins: round 1 is open and selected."""
    c = api()
    tok, j = join(c, {"total_rounds": 2})
    assert j["created_session"] is True and j["session"]["status"] == "active"
    v = poll(c, tok)
    assert v["round_number"] == 1 and v["phase"] == "train" and v["selected"] is True
    assert "expected_clients" not in v and "threshold" not in v


def test_fedavg_is_sample_weighted_mean_of_client_updates():
    c = api()
    t1, _ = join(c, FAST)
    t2, _ = join(c)
    warm(c, t1, t2)
    base = weight_codec.decode_weights(weights_for(c, t1))
    # round 1 belongs to the first client alone (the second one joined while it was running)
    assert submit(c, t1, 1, weight_codec.encode_weights(base)).json["aggregated"] is True
    assert poll(c, t1)["phase"] == "train" and poll(c, t2)["phase"] == "train"      # round 2: both
    up1 = [b + 1.0 for b in base]
    up2 = [b + 3.0 for b in base]
    submit(c, t1, 2, weight_codec.encode_weights(up1))
    assert poll(c, t1)["phase"] == "wait"
    submit(c, t2, 2, weight_codec.encode_weights(up2))
    new = weight_codec.decode_weights(weights_for(c, t1))
    assert np.allclose(new[1], base[1] + 2.0, atol=1e-5)       # equal shard sizes -> plain mean


def test_stale_round_and_duplicate_rejected_409():
    c = api()
    tok, _ = join(c, FAST)
    w = weights_for(c, tok)
    poll(c, tok)
    assert submit(c, tok, 5, w).status_code == 409
    assert submit(c, tok, 1, w).status_code == 200
    r = submit(c, tok, 1, w)
    assert r.status_code == 409


# ── fixed settings: the browser cannot change them ───────────────────────

def test_training_settings_are_fixed_by_the_server():
    from server import training_settings as ts
    c = api()
    evil = {
        "total_rounds": 4, "learning_rate": 0.9, "local_epochs": 5, "batch_size": 8,
        "round_timeout_seconds": 30, "min_participation_threshold": 3, "clients_per_round": 1,
        "num_clients": 9, "selection_strategy": "random",
    }
    _, j = join(c, evil)
    cfg = j["session"]["config"]
    assert cfg["total_rounds"] == 4                                   # the person may choose the rounds
    assert cfg["learning_rate"] == ts.LEARNING_RATE and cfg["local_epochs"] == ts.LOCAL_EPOCHS
    assert cfg["batch_size"] == ts.BATCH_SIZE and cfg["round_timeout_seconds"] == ts.ROUND_TIMEOUT_SECONDS
    assert cfg["min_participation_threshold"] == 1 and cfg["selection_strategy"] == "auto"
    assert "num_clients" not in cfg and "clients_per_round" not in cfg


# ── multi-client: anybody can join at any time ───────────────────────────

def test_three_language_clients_get_unique_languages_and_train_together():
    c = api()
    toks = [join(c, {**FAST, "partition_type": "language"})[0] for _ in range(3)]
    langs = {x["language"] for x in c.get("/api/clients").json["clients"]}
    assert langs == {"Dholuo", "Kalenjin", "Kidawida"}
    warm(c, *toks)
    assert submit(c, toks[0], 1, weights_for(c, toks[0])).json["aggregated"] is True   # round 1: first client alone
    for t in toks:
        assert poll(c, t)["phase"] == "train"                                          # round 2: everybody
    w = weights_for(c, toks[0])
    for t in toks[:-1]:
        assert submit(c, t, 2, w).json["aggregated"] is False
    assert submit(c, toks[-1], 2, w).json["aggregated"] is True
    shard = c.get("/api/dataset/shard", headers=H(toks[1])).json
    assert shard["num_samples"] == 2000 and [k for k, v in shard["label_distribution"].items() if v] == ["Kalenjin"]
    # there is no fixed number of clients: a 4th device simply joins
    t4, j4 = join(c)
    assert j4["created_session"] is False and j4["client"]["language"] == "Dholuo"


def test_concurrent_submissions_aggregate_exactly_once():
    c = api()
    toks = [join(c, {"total_rounds": 3})[0] for _ in range(3)]
    warm(c, *toks)
    w = weights_for(c, toks[0])
    assert submit(c, toks[0], 1, w).json["aggregated"] is True
    for t in toks:
        assert poll(c, t)["phase"] == "train"
    out = []

    def go(t):
        out.append(submit(app.test_client(), t, 2, w).status_code)

    threads = [threading.Thread(target=go, args=(t,)) for t in toks]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert out.count(200) == 3
    rounds = c.get("/api/metrics/rounds").json["rounds"]
    assert [r["round"] for r in rounds] == [1, 2]               # round 2 aggregated exactly once
    assert poll(c, toks[0])["round_number"] == 3


def test_late_client_joins_running_session_and_inherits_settings():
    c = api()
    t1, j1 = join(c, {"total_rounds": 5, "partition_type": "dirichlet"})
    assert poll(c, t1)["round_number"] == 1
    t2, j2 = join(c, {"total_rounds": 99, "partition_type": "iid"})
    assert j2["created_session"] is False and j2["session"]["session_id"] == j1["session"]["session_id"]
    assert j2["session"]["config"]["partition_type"] == "dirichlet"          # data split belongs to the training
    assert j2["client"]["rounds_wanted"] == 99 and j1["client"]["rounds_wanted"] == 5   # rounds are PER CLIENT
    assert j2["client"]["partition_index"] != j1["client"]["partition_index"]


# ── rounds are per client ────────────────────────────────────────────────

def test_every_client_gets_exactly_the_rounds_it_asked_for():
    """A asks for 6 rounds, B joins at round 3 asking for 2, C joins at round 3 asking for 3."""
    c = api()
    a, ja = join(c, {"total_rounds": 6})
    warm(c, a)
    w = weights_for(c, a)
    for rn in (1, 2):
        assert submit(c, a, rn, w).json["aggregated"] is True
    b, jb = join(c, {"total_rounds": 2})
    d, jd = join(c, {"total_rounds": 3})
    assert jb["session"]["config"]["total_rounds"] == 6 and jb["client"]["rounds_wanted"] == 2 and jb["client"]["joined_round"] == 3
    warm(c, b, d)
    assert poll(c, a)["total_rounds"] == 6 and poll(c, b)["total_rounds"] == 6        # planned end of the whole training
    assert submit(c, a, 3, w).json["aggregated"] is True                              # round 3: A alone
    taken = {}
    for rn in (4, 5, 6):
        views = {n: poll(c, t) for n, t in (("a", a), ("b", b), ("d", d))}
        taken[rn] = sorted(n for n, v in views.items() if v["phase"] == "train")
        for n, t in (("a", a), ("b", b), ("d", d)):
            if views[n]["phase"] == "train":
                assert submit(c, t, rn, w).status_code == 200
    assert taken == {4: ["a", "b", "d"], 5: ["a", "b", "d"], 6: ["a", "d"]}, taken
    # B did exactly 2 rounds, D exactly 3, A exactly 6; the training ended after A's 6th round
    ov = c.get("/api/sessions/current").json
    done = {x["client_name"]: x["rounds_done"] for x in ov["clients"]}
    assert sorted(done.values()) == [2, 3, 6]
    assert ov["session"]["status"] == "completed" and ov["summary"]["rounds_completed"] == 6


def test_every_client_keeps_its_own_data_partitioning():
    """A starts with IID, B joins with language, C with dirichlet: nobody inherits A's partitioning."""
    c = api()
    a, ja = join(c, {"total_rounds": 3, "partition_type": "iid"})
    b, jb = join(c, {"total_rounds": 3, "partition_type": "language"})
    d, jd = join(c, {"total_rounds": 3, "partition_type": "dirichlet", "dirichlet_alpha": 0.3})
    assert [x["client"]["partition_type"] for x in (ja, jb, jd)] == ["iid", "language", "dirichlet"]
    assert jb["client"]["language"] and not ja["client"]["language"] and not jd["client"]["language"]
    sa, sb, sd = (c.get("/api/dataset/shard", headers=H(t)).json for t in (a, b, d))
    assert (sa["partition_type"], sb["partition_type"], sd["partition_type"]) == ("iid", "language", "dirichlet")
    assert len(sb["label_distribution"]) and sum(1 for v in sb["label_distribution"].values() if v) == 1      # one language only
    assert sum(1 for v in sa["label_distribution"].values() if v) == 3                                         # IID: all three
    # a joiner that sends no partitioning inherits the session's
    e, je = join(c, {"total_rounds": 1})
    assert je["client"]["partition_type"] == "iid"


def test_results_global_totals_and_each_client_sees_only_its_own():
    """A asks 4 rounds, B joins and asks 2: global 'rounds' = client-rounds (6); each client's page shows only its own rounds."""
    c = api()
    a, ja = join(c, {"total_rounds": 4})
    warm(c, a)
    w = weights_for(c, a)
    assert submit(c, a, 1, w).json["aggregated"] is True
    b, jb = join(c, {"total_rounds": 2})
    warm(c, b)
    assert submit(c, a, 2, w).json["aggregated"] is True            # B joined during round 2 -> starts with round 3
    for rn in (3, 4):
        assert submit(c, a, rn, w).status_code == 200
        assert submit(c, b, rn, w).status_code == 200
    ga = c.get("/api/sessions/current?client_id=" + ja["client"]["client_id"]).json
    gb = c.get("/api/sessions/current?client_id=" + jb["client"]["client_id"]).json
    assert ga["summary"]["client_rounds"] == gb["summary"]["client_rounds"] == 6          # 4 + 2, same for everybody
    assert ga["summary"]["client_rounds_planned"] == 6 and ga["summary"]["aggregation_rounds"] == 4
    assert [r["round"] for r in ga["my"]["rounds"]] == [1, 2, 3, 4] and ga["my"]["summary"]["rounds_done"] == 4
    assert [r["round"] for r in gb["my"]["rounds"]] == [3, 4] and gb["my"]["summary"]["rounds_wanted"] == 2
    assert gb["my"]["client"]["client_id"] == jb["client"]["client_id"]
    assert all(r["global_accuracy"] is not None for r in gb["my"]["rounds"])
    # every client has its OWN per-language result for each of its rounds (not the shared one)
    assert all(r["per_class"] and len(r["per_class"]) == 3 and r["val_accuracy"] is not None for r in ga["my"]["rounds"] + gb["my"]["rounds"])
    assert c.get("/api/sessions/current").json["my"] is None


def test_client_that_finished_its_rounds_is_not_selected_and_session_continues():
    c = api()
    a, _ = join(c, {"total_rounds": 5})
    b, _ = join(c, {"total_rounds": 1})
    warm(c, a, b)
    w = weights_for(c, a)
    assert submit(c, a, 1, w).json["aggregated"] is True
    assert submit(c, b, 2, w).status_code == 200 and submit(c, a, 2, w).json["aggregated"] is True    # B's only round (round 2)
    v = poll(c, b)
    assert v["phase"] == "my_rounds_done" and v["my_rounds_done"] == 1 and v["session_status"] == "active"
    assert poll(c, a)["phase"] == "train" and poll(c, a)["selected_count"] == 1                          # A carries on alone
    assert c.post("/api/clients/leave", headers=H(b), json={}).status_code == 200
    assert poll(c, a)["session_status"] == "active"                                                       # B leaving does not end it
    sel = [s for s in c.get("/api/selection").json["selection"] if s["round"] == 3]
    assert any("finished its own rounds" in s["reason"] for s in sel)


def test_late_joiner_never_delays_the_running_round_and_starts_next_round():
    c = api()
    a, _ = join(c, FAST)
    warm(c, a)
    b, _ = join(c)
    vb = poll(c, b)
    assert vb["round_number"] == 1 and vb["phase"] == "not_selected" and vb["selected"] is False
    # A is NOT held up by B: its single update completes round 1 immediately
    assert submit(c, a, 1, weights_for(c, a)).json["aggregated"] is True
    assert poll(c, a)["phase"] == "train" and poll(c, b)["phase"] == "train" and poll(c, b)["round_number"] == 2
    assert [r["client_count"] for r in c.get("/api/metrics/rounds").json["rounds"]] == [1]


def test_joiner_that_is_still_loading_is_not_put_into_a_round():
    c = api()
    a, _ = join(c, {"total_rounds": 4})
    warm(c, a)
    b, _ = join(c)                                             # joined but has NOT polled yet (still loading model/data)
    assert submit(c, a, 1, weights_for(c, a)).json["aggregated"] is True
    assert poll(c, a)["selected_count"] == 1                   # round 2 = A only; nobody waits for the loading browser
    sel = [s for s in c.get("/api/selection").json["selection"] if s["round"] == 2]
    assert any(s["reason"].startswith("still loading") for s in sel)
    assert poll(c, b)["phase"] == "not_selected"               # B is ready now ...
    assert submit(c, a, 2, weights_for(c, a)).json["aggregated"] is True
    assert poll(c, b)["phase"] == "train" and poll(c, a)["selected_count"] == 2      # ... and takes part from round 3


def test_session_cap_protects_the_server(monkeypatch):
    from server import training_settings as ts
    monkeypatch.setattr(ts, "MAX_LIVE_CLIENTS", 2)
    c = api()
    join(c, FAST)
    join(c)
    r = c.post("/api/clients/join", json={"capabilities": {"backend": "cpu"}})
    assert r.status_code == 409 and r.json["error"] == "session_full"


# ── the training that can be joined ──────────────────────────────────────

def test_joinable_lists_the_running_training():
    c = api()
    assert c.get("/api/sessions/joinable").json["sessions"] == []
    join(c, {"total_rounds": 5, "partition_type": "language"})
    rows = c.get("/api/sessions/joinable").json["sessions"]
    assert len(rows) == 1
    r = rows[0]
    assert r["total_rounds"] == 5 and r["live_clients"] == 1 and r["partition_type"] == "language"
    assert r["rounds_completed"] == 0 and "capacity" not in r and "num_clients" not in r


def test_mode_new_starts_independent_second_session():
    c = api()
    a, ja = join(c, {"total_rounds": 5})
    b, jb = join(c, {"total_rounds": 2}, mode="new")
    assert jb["created_session"] is True and jb["session"]["session_id"] != ja["session"]["session_id"]
    assert jb["session"]["config"]["total_rounds"] == 2 and ja["session"]["config"]["total_rounds"] == 5
    assert c.get(f"/api/sessions/{ja['session']['session_id']}").json["session"]["status"] in ("pending", "active")
    assert len(c.get("/api/sessions/joinable").json["sessions"]) == 2
    assert poll(c, b)["total_rounds"] == 2


def test_mode_join_attaches_to_chosen_session_and_ignores_own_partition():
    c = api()
    a, ja = join(c, {"total_rounds": 5, "partition_type": "language"})
    b, jb = join(c, {"total_rounds": 99, "partition_type": "iid"}, mode="join", session_id=ja["session"]["session_id"])
    assert jb["created_session"] is False
    assert jb["session"]["session_id"] == ja["session"]["session_id"] and jb["session"]["config"]["partition_type"] == "language"
    assert jb["client"]["partition_index"] != ja["client"]["partition_index"]


def test_mode_join_unknown_or_missing_session_is_rejected():
    c = api()
    join(c, FAST)
    base = {"capabilities": {"backend": "cpu"}}
    r = c.post("/api/clients/join", json={**base, "mode": "join", "session_id": "does-not-exist"})
    assert r.status_code == 409 and r.json["error"] == "session_unavailable"
    r = c.post("/api/clients/join", json={**base, "mode": "join"})
    assert r.status_code == 400 and r.json["field"] == "session_id"
    r = c.post("/api/clients/join", json={**base, "mode": "weird"})
    assert r.status_code == 400 and r.json["field"] == "mode"


def test_two_sessions_aggregate_independently():
    c = api()
    a, ja = join(c, {"total_rounds": 2})
    b, jb = join(c, {"total_rounds": 3}, mode="new")
    poll(c, a); poll(c, b)
    assert submit(c, a, 1, weights_for(c, a)).status_code == 200
    assert poll(c, a)["round_number"] == 2 and poll(c, b)["round_number"] == 1
    assert submit(c, b, 1, weights_for(c, b)).status_code == 200
    ra = c.get(f"/api/sessions/{ja['session']['session_id']}").json
    rb = c.get(f"/api/sessions/{jb['session']['session_id']}").json
    assert ra["session"]["current_round"] >= 1 and rb["session"]["current_round"] >= 1


# ── automatic client selection ───────────────────────────────────────────

def test_every_ready_client_takes_part_and_the_choice_is_logged():
    c = api()
    toks = [join(c, FAST)[0] for _ in range(3)]
    warm(c, *toks)
    assert submit(c, toks[0], 1, weights_for(c, toks[0])).json["aggregated"] is True
    views = [poll(c, t) for t in toks]
    assert all(v["phase"] == "train" for v in views) and views[0]["selected_count"] == 3
    sel = [s for s in c.get("/api/selection").json["selection"] if s["round"] == 2]
    assert len(sel) == 3 and all(s["selected"] and s["strategy"] == "auto_all" and s["reason"] for s in sel)


def test_over_the_per_round_limit_server_picks_by_capability_plus_fairness_place():
    c = api()
    weak, _ = join(c, {"total_rounds": 5}, caps={"backend": "cpu", "cpu_cores": 1, "memory_gb": 1})
    strong, _ = join(c, caps={"backend": "webgpu", "cpu_cores": 16, "memory_gb": 16})
    mid, _ = join(c, caps={"backend": "webgl", "cpu_cores": 4, "memory_gb": 4})
    warm(c, weak, strong, mid)
    cfg_with(max_clients_per_round=2)
    assert submit(c, weak, 1, weights_for(c, weak)).json["aggregated"] is True        # round 1 = weak (first client)
    views = {n: poll(c, t) for n, t in (("weak", weak), ("strong", strong), ("mid", mid))}
    assert views["strong"]["phase"] == "train"                                         # best device always gets a place
    assert sum(v["selected"] for v in views.values()) == 2
    sel = [s for s in c.get("/api/selection").json["selection"] if s["round"] == 2]
    assert {s["strategy"] for s in sel} == {"auto_resource"}
    assert any(s["selected"] and s["reason"] == "highest capability score" for s in sel)
    assert any(s["selected"] and s["reason"].startswith("random draw") for s in sel)
    assert any(not s["selected"] and "limit" in s["reason"] for s in sel)


def test_paused_client_is_not_selected_and_resumes():
    c = api()
    a, _ = join(c, {"total_rounds": 5})
    b, _ = join(c)
    warm(c, a, b)
    assert c.post("/api/clients/status", headers=H(b), json={"state": "paused"}).status_code == 200
    assert submit(c, a, 1, weights_for(c, a)).json["aggregated"] is True
    assert poll(c, a)["phase"] == "train" and poll(c, b)["phase"] == "not_selected"      # round 2 without B
    assert c.post("/api/clients/status", headers=H(b), json={"state": "bogus"}).status_code == 400
    assert c.post("/api/clients/status", headers=H(b), json={"state": "active"}).status_code == 200
    assert submit(c, a, 2, weights_for(c, a)).json["aggregated"] is True
    assert poll(c, b)["phase"] == "train"                                                  # back in from round 3


def test_single_client_pause_and_resume_keeps_the_session_alive():
    c = api()
    a, _ = join(c, FAST)
    poll(c, a)
    assert c.post("/api/clients/status", headers=H(a), json={"state": "paused"}).status_code == 200
    # the round that was already being trained is still accepted ...
    assert submit(c, a, 1, weights_for(c, a)).json["aggregated"] is True
    v = poll(c, a)
    assert v["session_status"] == "active" and v["round_number"] == 2 and v["phase"] == "not_selected"
    # ... and the session is NOT aborted while the only client stays paused, even past the round timeout
    _expire_round()
    v = poll(c, a)
    assert v["session_status"] == "active" and v["round_number"] == 2
    assert c.post("/api/clients/status", headers=H(a), json={"state": "active"}).status_code == 200
    v = poll(c, a)
    assert v["phase"] == "train" and v["selected"] is True
    assert submit(c, a, 2, weights_for(c, a)).json["aggregated"] is True


# ── timeout / exclusion / recovery ───────────────────────────────────────

def _expire_round(rn=None):
    with app.app_context():
        for r in Round.query.filter(Round.state.in_(("open", "collecting"))).all():
            r.deadline = utcnow() - timedelta(seconds=1)
        db.session.commit()


def _two_clients_in_round_two():
    c = api()
    a, _ = join(c, {"total_rounds": 5})
    b, _ = join(c)
    warm(c, a, b)
    w = weights_for(c, a)
    assert submit(c, a, 1, w).json["aggregated"] is True
    assert poll(c, b)["phase"] == "train"
    return c, a, b, w


def test_timeout_aggregates_with_threshold_and_excludes_silent_client():
    c, a, b, w = _two_clients_in_round_two()
    submit(c, a, 2, w)
    assert poll(c, a)["round_number"] == 2                      # B is live and working -> wait for B (only)
    _expire_round()
    assert poll(c, a)["round_number"] == 3                      # timed out -> aggregated with 1 of 2
    rounds = c.get("/api/metrics/rounds").json["rounds"]
    assert [r["client_count"] for r in rounds] == [1, 1]
    assert any(e["event_type"] == "client_dropped" for e in c.get("/api/events").json["events"])


def test_dead_client_does_not_block_round():
    c, a, b, w = _two_clients_in_round_two()
    with app.app_context():                                      # B vanished 10 minutes ago
        cl = Client.query.filter_by(session_token=b).first()
        cl.last_seen = utcnow() - timedelta(minutes=10)
        db.session.commit()
    assert submit(c, a, 2, w).json["aggregated"] is True


def test_round_that_nobody_finishes_is_retried_then_session_aborts():
    c = api()
    a, _ = join(c, FAST)
    poll(c, a)
    for attempt in (1, 2):
        _expire_round()
        v = poll(c, a)
        assert v["session_status"] == "active" and v["round_number"] == 1
    _expire_round()
    v = poll(c, a)
    assert v["session_status"] == "aborted" and v["phase"] == "aborted"


def test_other_client_takes_over_when_the_selected_one_disappears():
    c = api()
    a, _ = join(c, FAST)
    warm(c, a)
    b, _ = join(c)
    warm(c, b)                                                   # B waits for the next round
    with app.app_context():                                      # A's tab crashed
        Client.query.filter_by(session_token=a).first().last_seen = utcnow() - timedelta(minutes=10)
        db.session.commit()
    v = poll(c, b)                                               # nothing to wait for: B is handed the round
    assert v["phase"] == "train" and v["selected"] is True and v["session_status"] == "active"
    assert submit(c, b, 1, weights_for(c, b)).json["aggregated"] is True


def test_state_survives_restart_because_it_lives_in_the_database():
    c = api()
    tok, _ = join(c, FAST)
    w = weights_for(c, tok)
    poll(c, tok)
    submit(c, tok, 1, w)
    fresh = app.test_client()                                    # brand-new client/"worker"
    v = poll(fresh, tok)
    assert v["round_number"] == 2 and v["model_version"] == 1
    assert fresh.get("/api/model/global", headers=H(tok)).json["version"] == 1


def test_stuck_aggregating_round_is_recovered():
    c = api()
    tok, _ = join(c, FAST)
    w = weights_for(c, tok)
    poll(c, tok)
    with app.app_context():
        r = Round.query.first()
        r.state = "aggregating"
        r.aggregating_since = utcnow() - timedelta(minutes=5)
        db.session.add(ClientUpdate(session_id=r.session_id, round_number=1, client_id=Client.query.first().client_id,
                                    dataset_size=2000, weights_blob=weight_codec.pack_blob(weight_codec.decode_weights(w))))
        db.session.commit()
    assert poll(c, tok)["round_number"] == 2


def test_new_start_after_everyone_left_creates_new_session():
    c = api()
    tok, j1 = join(c, FAST)
    assert c.post("/api/clients/leave", headers=H(tok), json={}).status_code == 200
    assert c.get("/api/rounds/current", headers=H(tok)).status_code == 401
    _, j2 = join(c, FAST)
    assert j2["created_session"] is True and j2["session"]["session_id"] != j1["session"]["session_id"]


# ── metrics / export / timestamps / global summary ───────────────────────

def test_timestamps_are_utc_with_milliseconds():
    c = api()
    tok, _ = join(c, FAST)
    poll(c, tok)
    submit(c, tok, 1, weights_for(c, tok))
    ts = c.get("/api/metrics/latest").json["latest"]["timestamp"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", ts)


def test_exports_csv_and_json():
    c = api()
    tok, _ = join(c, {"total_rounds": 1})
    poll(c, tok)
    submit(c, tok, 1, weights_for(c, tok))
    csv_ = c.get("/api/export/csv")
    assert csv_.status_code == 200 and csv_.data.decode().splitlines()[0].startswith("session_id,round,timestamp,accuracy")
    assert len(csv_.data.decode().strip().splitlines()) == 2
    assert "local_accuracy" in c.get("/api/export/clients.csv").data.decode()
    js = c.get("/api/export/json").json
    assert js["rounds"][0]["per_class"] and js["selection_log"]


def test_dashboard_overview_contains_everything_the_ui_needs():
    c = api()
    tok, _ = join(c, FAST)
    poll(c, tok)
    ov = c.get("/api/sessions/current").json
    assert ov["session"]["status"] == "active" and ov["clients"][0]["online"] is True
    assert ov["model"]["parameters"] == 160579 and ov["partition_indicator"] == "iid"
    assert ov["summary"]["clients_online"] == 1 and ov["summary"]["accuracy"] is None and ov["summary"]["rounds_completed"] == 0


def test_global_summary_combines_all_clients_of_the_training():
    c = api()
    toks = [join(c, {"total_rounds": 4}, caps={"backend": b, "cpu_cores": 4, "memory_gb": 8})[0]
            for b in ("webgpu", "cpu", "webgl")]            # every client asks for 4 rounds
    warm(c, *toks)
    w = weights_for(c, toks[0])
    submit(c, toks[0], 1, w)                                    # round 1: first client
    for t in toks:
        submit(c, t, 2, w)                                      # round 2: all three
    s = c.get("/api/sessions/current").json["summary"]          # no session id, no token: any viewer on any device
    assert s["rounds_completed"] == 2 and s["total_rounds"] == 5 and s["current_round"] == 3          # A planned 4, the two who joined in round 1 plan to reach 5
    assert s["clients_total"] == 3 and s["clients_contributed"] == 3 and s["clients_last_round"] == 3
    assert s["clients_online"] == 3 and s["devices"] == ["cpu", "webgl", "webgpu"]
    assert 0 <= s["accuracy"] <= 1 and s["f1"] is not None and s["precision"] is not None
    latest = c.get("/api/metrics/latest").json["latest"]
    assert s["accuracy"] == latest["accuracy"] and latest["client_count"] == 3


def test_results_page_follows_the_running_training_then_falls_back_to_the_last_finished_one():
    c = api()
    a, ja = join(c, {"total_rounds": 1})
    submit(c, a, 1, weights_for(c, a))                          # training 1 completes (has results)
    assert poll(c, a)["session_status"] == "completed"
    c.post("/api/clients/leave", headers=H(a), json={})
    b, jb = join(c, FAST)                                       # training 2 starts: it is the one shown
    cur = c.get("/api/sessions/current").json
    assert cur["session"]["session_id"] == jb["session"]["session_id"]
    c.post("/api/clients/leave", headers=H(b), json={})         # training 2 ends without results ...
    cur = c.get("/api/sessions/current").json
    assert cur["session"]["session_id"] == ja["session"]["session_id"] and cur["summary"]["rounds_completed"] == 1


def test_data_shards_for_any_number_of_clients():
    from server import data_service as ds
    sets = [set(ds._iid_indices(i, 42)) for i in range(6)]
    assert all(len(s) == 2000 for s in sets)
    assert len(sets[0] | sets[1] | sets[2]) == 6000 and not (sets[0] & sets[1])      # first three: disjoint thirds
    assert sets[3] != sets[0]                                                          # next block: re-shuffled
    _, y = ds.client_shard("iid", 0, "", 0.5, 42)
    assert max(ds.shard_label_distribution(y).values()) < 800                           # mixed languages
    for i in range(5):
        x, y = ds.client_shard("dirichlet", i, "", 0.5, 42)
        assert len(y) == 2000 and x.shape == (2000, 40)
    assert ds.shard_label_distribution(ds.client_shard("language", 4, "Kalenjin", 0.5, 42)[1])["Kalenjin"] == 2000
    a = ds.client_shard("dirichlet", 2, "", 0.5, 42)[1]
    assert (a == ds.client_shard("dirichlet", 2, "", 0.5, 42)[1]).all()               # deterministic


def test_destructive_legacy_endpoints_are_locked_without_admin_key():
    assert api().delete("/experiment/reset").status_code == 403
    assert api().delete("/clients/reset").status_code == 403


def test_legacy_global_model_format_is_loadable():
    c = api()
    m = c.get("/global-model").json
    assert m["weights"][0]["shape"] == [5000, 32] and len(m["weights"][0]["data"]) == 160000
