"""
server/fl_service.py
====================
Federated-learning coordination engine (SDS 3.1.2 / 3.2.1, SRS US-07..US-13,
UC-10b, NFR-R01/R05/R07).

DESIGN RULES
  * ALL state lives in the database (sessions, rounds, updates, checkpoints).
    Nothing is kept in process memory, so it is correct with several gunicorn
    workers/threads and survives a Render restart (NFR-R01/R05/R07).
  * No background thread is needed: round timeouts and the aggregation itself
    are triggered lazily by the clients' polling calls.  Aggregation is
    claimed with an atomic `UPDATE ... WHERE state IN (open, collecting)` so
    it runs exactly once even if two requests race.
  * Round state machine (SDS): open -> collecting -> aggregating -> complete
    (+ failed when the participation threshold could not be met).

HOW CLIENTS TAKE PART (v3)
  * There is ONE shared training at a time.  The first client that presses
    "Start Training" creates it (its round count / data split are used); every
    later client that presses "Start Training" simply joins it - no expected
    number of clients, no waiting for anybody.
  * A newcomer is put into the NEXT round, never into the one that is already
    running, and only once its browser has loaded the model and its data
    ("ready").  So clients that are already training never wait for a newcomer.
  * A round is aggregated as soon as every client that is really working on it
    has submitted (or the fixed round timeout ran out).  One client is enough.
  * "Rounds" is PER CLIENT: every client takes part in exactly the number of
    rounds it asked for, then finishes while the others carry on.  The training
    ends when no online client wants another round.
  * Learning rate / epochs / batch size / timeout / client selection are fixed
    in server/training_settings.py and cannot be changed from the browser.
"""

import json
import random
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import text

from server import data_service, model_eval, model_spec, training_settings, weight_codec
from server.api_errors import ApiError, bad_request, conflict, forbidden, not_found
from server.database import db
from server.fedavg import fedavg_aggregate
from server.models import (
    Client,
    ClientEvent,
    ClientSelectionLog,
    ClientUpdate,
    ExperimentLog,
    ExperimentSession,
    ModelCheckpoint,
    Round,
    iso,
    utcnow,
)

# A client that has not contacted the server for this long is considered gone.
LIVE_SECONDS = 75
# A session that nobody has joined for this long (while pending) is abandoned.
PENDING_GRACE_SECONDS = 60
# A round stuck in "aggregating" (worker crashed) is re-claimed after this.
AGGREGATING_STALE_SECONDS = 60
MAX_ROUND_ATTEMPTS = 3
KEEP_CHECKPOINTS_WITH_WEIGHTS = 3

DEFAULT_CONFIG = training_settings.DEFAULT_CONFIG
# the only settings the person who starts a training may choose
USER_SETTINGS = tuple(training_settings.USER_DEFAULTS)

_join_lock = threading.Lock()


# ── helpers ──────────────────────────────────────────────────────────────

def _int(cfg, key, lo, hi, default):
    v = cfg.get(key, default)
    try:
        if isinstance(v, bool):
            raise ValueError
        v = int(v)
    except (TypeError, ValueError):
        raise bad_request(f"'{key}' must be an integer", key)
    if not lo <= v <= hi:
        raise bad_request(f"'{key}' must be between {lo} and {hi}", key)
    return v


def _float(cfg, key, lo, hi, default):
    v = cfg.get(key, default)
    try:
        if isinstance(v, bool):
            raise ValueError
        v = float(v)
    except (TypeError, ValueError):
        raise bad_request(f"'{key}' must be a number", key)
    if not (lo <= v <= hi):
        raise bad_request(f"'{key}' must be between {lo} and {hi}", key)
    return v


def _wanted_rounds(raw: Optional[dict]) -> int:
    """The number of rounds one client asked for (its own choice, 1-100)."""
    raw = dict(raw or {})
    v = raw.get("total_rounds", raw.get("num_rounds", raw.get("totalRounds")))
    return _int({"total_rounds": v if v is not None else DEFAULT_CONFIG["total_rounds"]}, "total_rounds", 1, 100, 20)


def rounds_done_map(session_id: str) -> Dict[str, int]:
    """client_id -> number of rounds that client has contributed an update to."""
    rows = (
        db.session.query(ClientUpdate.client_id, db.func.count(ClientUpdate.id))
        .filter(ClientUpdate.session_id == session_id)
        .group_by(ClientUpdate.client_id)
        .all()
    )
    return {cid: int(n) for cid, n in rows}


def _client_part(c: Client, cfg: dict):
    """(partition_type, dirichlet_alpha) of ONE client - its own choice, else the session's."""
    return (c.partition_type or cfg["partition_type"], float(c.dirichlet_alpha if c.dirichlet_alpha is not None else cfg["dirichlet_alpha"]))


def _own_partition(raw: Optional[dict], cfg: dict):
    """The partitioning a joining client asked for (falls back to the session's when it sent none)."""
    raw = dict(raw or {})
    merged = {k: raw.get(k, raw.get(a)) for k, a in (("partition_type", "partitionType"), ("dirichlet_alpha", "alpha"))}
    merged = {k: v for k, v in merged.items() if v is not None}
    n = normalize_config({**{"partition_type": cfg["partition_type"], "dirichlet_alpha": cfg["dirichlet_alpha"]}, **merged})
    return n["partition_type"], n["dirichlet_alpha"]


def _wanted(c: Client, cfg: dict) -> int:
    return int(c.rounds_wanted or cfg["total_rounds"])


def planned_rounds(sess: ExperimentSession, clients: Optional[List[Client]] = None, done: Optional[Dict[str, int]] = None) -> int:
    """Where the training is expected to end: the furthest round any (still interested) client plans to reach."""
    if sess.status == "completed":
        return max(sess.current_round, 1)
    cfg = sess.config()
    clients = clients if clients is not None else Client.query.filter_by(session_id=sess.session_id).all()
    done = done if done is not None else rounds_done_map(sess.session_id)
    ends = [
        (c.joined_round or 0) + _wanted(c, cfg)
        for c in clients
        if c.status != "left" or done.get(c.client_id, 0) >= _wanted(c, cfg)
    ]
    return max([sess.current_round, *ends]) if ends else max(sess.current_round, cfg["total_rounds"])


def normalize_config(raw: Optional[dict]) -> dict:
    """
    Validate the settings a person may choose when starting a training
    (rounds, data partitioning).  Everything else - learning rate, epochs,
    batch size, timeout, threshold, client selection - is fixed by the server
    (server/training_settings.py); such keys sent by a client are ignored.
    """
    raw = dict(raw or {})
    cfg = dict(DEFAULT_CONFIG)
    # accept the legacy camelCase / alias names used by the first frontend
    aliases = {
        "num_rounds": "total_rounds",
        "totalRounds": "total_rounds",
        "partitionType": "partition_type",
        "alpha": "dirichlet_alpha",
    }
    for old, new in aliases.items():
        if old in raw and new not in raw:
            raw[new] = raw[old]
    if raw.get("partition_type") == "non_iid":
        raw["partition_type"] = "dirichlet"
    if raw.get("aggregation_method") not in (None, "fedavg"):
        raise bad_request(
            "only 'fedavg' is available (FedProx is future work in the SRS/SDS)", "aggregation_method"
        )
    cfg.update({k: raw[k] for k in USER_SETTINGS if raw.get(k) is not None})

    cfg["total_rounds"] = _int(cfg, "total_rounds", 1, 100, 20)
    cfg["dirichlet_alpha"] = _float(cfg, "dirichlet_alpha", 0.05, 10.0, 0.5)
    if cfg["partition_type"] not in ("language", "iid", "dirichlet"):
        raise bad_request("'partition_type' must be language, iid or dirichlet", "partition_type")
    return cfg


def event(session_id, client_id, event_type, detail=""):
    db.session.add(
        ClientEvent(session_id=session_id, client_id=client_id, event_type=event_type, detail=(detail or "")[:300])
    )


def capability_score(cpu_cores, memory_gb, backend) -> float:
    """0..1 resource score used by resource-aware selection (SDS 3.1.2.2)."""
    cores = min(max(int(cpu_cores or 1), 1), 16) / 16.0
    mem = min(max(float(memory_gb or 1), 0.25), 16.0) / 16.0
    be = {"webgpu": 1.0, "webgl": 0.6, "wasm": 0.35, "cpu": 0.2}.get((backend or "cpu"), 0.2)
    return round(0.4 * cores + 0.3 * mem + 0.3 * be, 4)


# ── sessions ─────────────────────────────────────────────────────────────

def get_session(session_id: str) -> Optional[ExperimentSession]:
    return db.session.get(ExperimentSession, session_id)


def latest_session() -> Optional[ExperimentSession]:
    """
    The training the dashboards / exports / status bars show when no explicit
    session id is given:
      1. the training that is running right now (newest, with live clients), else
      2. the most recent training that produced results, else
      3. the newest session of any kind.
    Read-only: it never closes anything.
    """
    running = (
        ExperimentSession.query.filter(ExperimentSession.status.in_(("pending", "active")))
        .order_by(ExperimentSession.created_at.desc())
        .all()
    )
    for sess in running:
        fresh = (utcnow() - sess.created_at).total_seconds() < PENDING_GRACE_SECONDS
        if fresh or live_clients(sess.session_id):
            return sess
    row = db.session.query(ExperimentLog.session_id).order_by(ExperimentLog.logged_at.desc()).first()
    if row and row[0]:
        found = db.session.get(ExperimentSession, row[0])
        if found is not None:
            return found
    return ExperimentSession.query.order_by(ExperimentSession.created_at.desc()).first()


def live_clients(session_id: str, include_paused=True) -> List[Client]:
    cutoff = utcnow() - timedelta(seconds=LIVE_SECONDS)
    statuses = ("active", "paused") if include_paused else ("active",)
    return (
        Client.query.filter(Client.session_id == session_id, Client.status.in_(statuses), Client.last_seen >= cutoff)
        .order_by(Client.partition_index)
        .all()
    )


def _close_session(sess: ExperimentSession, status: str, reason: str):
    sess.status = status
    sess.ended_at = utcnow()
    event(sess.session_id, None, f"session_{status}", reason)


def list_joinable_sessions() -> List[ExperimentSession]:
    """
    Every pending/active session a new client may join, newest first.

    Sessions that have no live client left (and are not a freshly created
    pending one) are closed here as a housekeeping step.
    """
    candidates = (
        ExperimentSession.query.filter(ExperimentSession.status.in_(("pending", "active")))
        .order_by(ExperimentSession.created_at.desc())
        .all()
    )
    joinable = []
    for sess in candidates:
        live = live_clients(sess.session_id)
        fresh = (utcnow() - sess.created_at).total_seconds() < PENDING_GRACE_SECONDS
        if live or (sess.status == "pending" and fresh):
            joinable.append(sess)
        else:
            _close_session(sess, "aborted", "no live clients")
    return joinable


def joinable_summary() -> List[dict]:
    """Small public description of the running training (training page: 'a training is running - join it')."""
    out = []
    for sess in list_joinable_sessions():
        cfg = sess.config()
        latest = _latest_log(sess.session_id)
        out.append(
            {
                "session_id": sess.session_id,
                "status": sess.status,
                "current_round": sess.current_round,
                "rounds_completed": ExperimentLog.query.filter_by(session_id=sess.session_id).count(),
                "total_rounds": planned_rounds(sess),
                "partition_type": cfg["partition_type"],
                "dirichlet_alpha": cfg["dirichlet_alpha"],
                "live_clients": len(live_clients(sess.session_id)),
                "accuracy": latest.accuracy if latest else None,
                "f1": latest.f1_macro if latest else None,
                "created_at": iso(sess.created_at),
            }
        )
    db.session.commit()
    return out


def create_session(cfg: dict) -> ExperimentSession:
    sess = ExperimentSession(
        session_id=str(uuid.uuid4()),
        status="pending",
        config_json=json.dumps(cfg),
        current_round=0,
        model_version=0,
    )
    db.session.add(sess)
    # Round 0 checkpoint: the SERVER-initialised global weights (SDS 3.2.1 step 5).
    db.session.add(
        ModelCheckpoint(
            session_id=sess.session_id,
            round_number=0,
            weights_blob=weight_codec.pack_blob(model_spec.init_weights(cfg["seed"])),
        )
    )
    event(sess.session_id, None, "session_created", json.dumps(cfg)[:290])
    return sess


def _activate_if_pending(sess: ExperimentSession):
    """pending -> active and open round 1 (atomic claim, so it happens exactly once)."""
    if sess.status != "pending":
        return
    claimed = ExperimentSession.query.filter_by(session_id=sess.session_id, status="pending").update(
        {"status": "active", "started_at": utcnow()}, synchronize_session=False
    )
    if claimed != 1:
        return
    db.session.refresh(sess)
    _open_round(sess, 1)


def join_session(config_in: Optional[dict], info: dict, mode: str = "auto", session_id: Optional[str] = None) -> dict:
    """
    Register a browser client (US-08, US-06 capability reporting) and attach
    it to the training.

    mode="auto" (default - what "Start Training" uses): join the running
                training, or start one from `config_in` when none is running.
                No other client has to be present and nothing is waited for.
    mode="join": join the running session `session_id` (inherits ITS settings).
    mode="new" : always start a new session from `config_in` (API / tests only).
    """
    if mode not in ("auto", "join", "new"):
        raise bad_request("mode must be 'auto', 'join' or 'new'", "mode")
    if mode == "join" and not session_id:
        raise bad_request("session_id is required when mode is 'join'", "session_id")

    with _join_lock:
        if db.engine.dialect.name == "postgresql":
            db.session.execute(text("SELECT pg_advisory_xact_lock(727275)"))

        wanted_rounds = _wanted_rounds(config_in)       # validated before anything is created
        sess = None
        created = False
        if mode != "new":
            joinable = list_joinable_sessions()
            if mode == "join":
                sess = next((x for x in joinable if x.session_id == session_id), None)
                if sess is None:
                    db.session.commit()
                    raise conflict(
                        "That session is no longer running. Please choose again.", "session_unavailable"
                    )
            elif joinable:
                sess = joinable[0]
        if sess is None:
            sess = create_session(normalize_config(config_in))
            created = True
        cfg = sess.config()

        live = live_clients(sess.session_id)
        if len(live) >= training_settings.MAX_LIVE_CLIENTS:
            db.session.rollback()
            raise conflict(
                f"This training already has {len(live)} devices connected (the limit is "
                f"{training_settings.MAX_LIVE_CLIENTS}). Please try again in a moment.",
                "session_full",
            )

        # The data shard follows from the join index: the lowest free index among the live clients.
        used = {c.partition_index for c in live}
        pt, alpha = _own_partition(config_in, cfg)
        idx = None
        pref = (info.get("preferred_language") or "").strip()
        if pt == "language" and pref in model_spec.LABELS:
            cand = model_spec.LABELS.index(pref)
            idx = next((i for i in range(cand, training_settings.MAX_LIVE_CLIENTS, model_spec.NUM_CLASSES) if i not in used), None)
        if idx is None:
            idx = next(i for i in range(training_settings.MAX_LIVE_CLIENTS) if i not in used)

        language = model_spec.LABELS[idx % model_spec.NUM_CLASSES] if pt == "language" else None
        partition_name = (
            data_service.LANGUAGE_TO_PARTITION.get(language) if language else f"{pt}_{idx}"
        )

        token = secrets.token_urlsafe(24)
        base_name = (str(info.get("client_name") or "").replace(",", " ").strip() or f"Client-{idx + 1}")[:80]
        name = base_name
        if Client.query.filter_by(client_name=name).first():
            name = f"{base_name}-{token[:4]}"

        backend = info.get("backend") if info.get("backend") in ("webgpu", "webgl", "wasm", "cpu") else "cpu"
        cores = _safe_int(info.get("cpu_cores"), 1, 256)
        mem = _safe_float(info.get("memory_gb"), 0.1, 1024)
        client = Client(
            client_id=str(uuid.uuid4()),
            client_name=name,
            partition=partition_name,
            status="active",
            session_id=sess.session_id,
            partition_index=idx,
            language=language,
            session_token=token,
            cpu_cores=cores,
            memory_gb=mem,
            backend=backend,
            webgpu_available=bool(info.get("webgpu_available")),
            webnn_available=bool(info.get("webnn_available")),
            capability_score=capability_score(cores, mem, backend),
            user_agent=str(info.get("user_agent") or "")[:200],
            last_seen=utcnow(),
            ready=False,
            rounds_wanted=wanted_rounds,
            partition_type=pt,
            dirichlet_alpha=alpha,
            joined_round=0 if created else int(sess.current_round or 0),
        )
        db.session.add(client)
        event(sess.session_id, client.client_id, "client_joined", f"{name} idx={idx} backend={backend} rounds={wanted_rounds}")
        db.session.flush()
        _activate_if_pending(sess)       # first client: the training starts right now, nobody is waited for
        db.session.commit()
        return {"session": sess, "client": client, "created_session": created}


def _safe_int(v, lo, hi):
    try:
        v = int(v)
        return v if lo <= v <= hi else None
    except (TypeError, ValueError):
        return None


def _safe_float(v, lo, hi):
    try:
        v = float(v)
        return v if lo <= v <= hi else None
    except (TypeError, ValueError):
        return None


def touch(client: Client):
    now = utcnow()
    if client.last_seen is None or (now - client.last_seen).total_seconds() > 4:
        client.last_seen = now
        db.session.commit()


def set_client_state(client: Client, state: str):
    if state not in ("active", "paused"):
        raise bad_request("state must be 'active' or 'paused'", "state")
    if client.status != state:
        client.status = state
        event(client.session_id, client.client_id, "client_" + ("paused" if state == "paused" else "resumed"))
    client.last_seen = utcnow()
    db.session.commit()


def leave_session(client: Client, reason="left"):
    client.status = "left"
    event(client.session_id, client.client_id, "client_left", reason)
    db.session.commit()
    sess = get_session(client.session_id)
    if sess and sess.status in ("pending", "active") and not live_clients(sess.session_id):
        _close_session(sess, "aborted", "all clients left")
        db.session.commit()


# ── rounds ───────────────────────────────────────────────────────────────

def latest_round(session_id: str, round_number: int) -> Optional[Round]:
    return (
        Round.query.filter_by(session_id=session_id, round_number=round_number).order_by(Round.id.desc()).first()
    )


def _select_clients(sess, cfg, round_number, attempt) -> List[Client]:
    """
    AUTOMATIC client selection (SRS US-12 / US-13) - nobody chooses it:

      * every ready, active client takes part (the normal case), or
      * when more clients are online than `max_clients_per_round`, the server
        picks them: most places by capability score (resource-aware, US-13), a
        few by random draw (random baseline, US-12) so weak devices still get
        a turn.
    A client that has not finished loading yet ("not ready") and a paused
    client are never put into a round; a newcomer therefore never makes the
    others wait.
    """
    live = live_clients(sess.session_id)
    done = rounds_done_map(sess.session_id)
    finished = {c.client_id for c in live if done.get(c.client_id, 0) >= _wanted(c, cfg)}   # got all the rounds it asked for
    active = [c for c in live if c.status == "active" and c.client_id not in finished]
    ready = [c for c in active if c.ready]
    eligible = ready or active          # nobody is ready yet (first client still loading) -> everybody active
    cap = int(cfg.get("max_clients_per_round") or training_settings.MAX_CLIENTS_PER_ROUND)
    rng = random.Random(f"{cfg['seed']}-{round_number}-{attempt}")

    reasons: Dict[str, str] = {}
    if len(eligible) <= cap:
        strategy = "auto_all"
        chosen = list(eligible)
        for c in chosen:
            reasons[c.client_id] = "all available clients take part"
    else:
        strategy = "auto_resource"
        n_fair = max(1, int(round(cap * training_settings.FAIRNESS_FRACTION)))
        shuffled = sorted(eligible, key=lambda c: c.client_id)
        rng.shuffle(shuffled)
        ranked = sorted(shuffled, key=lambda c: -(c.capability_score or 0))
        top, rest = ranked[: cap - n_fair], ranked[cap - n_fair:]
        fair = rng.sample(rest, min(n_fair, len(rest)))
        chosen = top + fair
        for c in top:
            reasons[c.client_id] = "highest capability score"
        for c in fair:
            reasons[c.client_id] = "random draw (fairness place)"

    chosen_ids = {c.client_id for c in chosen}
    for c in live:
        if c.client_id in chosen_ids:
            reason = reasons[c.client_id]
        elif c.client_id in finished:
            reason = "client finished its own rounds"
        elif c.status == "paused":
            reason = "client paused"
        elif not c.ready and ready:
            reason = "still loading - takes part from the next round"
        else:
            reason = "not drawn (per-round limit)"
        db.session.add(
            ClientSelectionLog(
                session_id=sess.session_id,
                round_number=round_number,
                client_id=c.client_id,
                client_name=c.client_name,
                strategy=strategy,
                selected=c.client_id in chosen_ids,
                capability_score=c.capability_score,
                reason=reason,
            )
        )
    return chosen


def _open_round(sess: ExperimentSession, round_number: int, attempt: int = 1) -> Optional[Round]:
    cfg = sess.config()
    chosen = _select_clients(sess, cfg, round_number, attempt)
    if not chosen and not _anyone_wants_more_rounds(sess):
        if live_clients(sess.session_id):
            _close_session(sess, "completed", f"every client finished its rounds ({round_number - 1} rounds)")
        else:
            _close_session(sess, "aborted", f"no active clients available for round {round_number}")
        return None
    # `chosen` is empty only when everybody who is online is paused: the round is opened EMPTY
    # and picks up the first client that resumes / joins (see process_round).
    now = utcnow()
    rnd = Round(
        session_id=sess.session_id,
        round_number=round_number,
        state="open",
        min_clients=max(1, min(cfg["min_participation_threshold"], len(chosen))),
        start_time=now,
        deadline=now + timedelta(seconds=cfg["round_timeout_seconds"]),
        selected_clients=",".join(c.client_id for c in chosen),
        attempt=attempt,
    )
    sess.current_round = round_number
    db.session.add(rnd)
    event(
        sess.session_id, None, "round_opened",
        f"round {round_number} attempt {attempt} selected={len(chosen)}" + ("" if chosen else " (everybody paused - waiting)"),
    )
    return rnd


def _anyone_wants_more_rounds(sess: ExperimentSession) -> bool:
    """True while at least one online client (active or paused) has not had all the rounds it asked for."""
    cfg = sess.config()
    done = rounds_done_map(sess.session_id)
    return any(done.get(c.client_id, 0) < _wanted(c, cfg) for c in live_clients(sess.session_id))


def _maybe_start_session(sess: ExperimentSession):
    """Safety net for a session still 'pending' (normally join_session already started it)."""
    if sess.status != "pending":
        return
    if not any(c.status == "active" for c in live_clients(sess.session_id)):
        return
    _activate_if_pending(sess)
    db.session.commit()


def _pending_live(rnd: Round, submitted: set) -> List[Client]:
    """Selected clients that are still online and have not submitted yet."""
    cutoff = utcnow() - timedelta(seconds=LIVE_SECONDS)
    pending = [c for c in rnd.selected_list() if c not in submitted]
    if not pending:
        return []
    return Client.query.filter(
        Client.client_id.in_(pending), Client.status.in_(("active", "paused")), Client.last_seen >= cutoff
    ).all()


def process_round(sess: ExperimentSession, rnd: Round):
    """
    Drive the round state machine.  Called from every poll / submit, so
    timeouts and exclusion of silent clients need no background worker.

    Rule: the round waits ONLY for clients that are actively training on it.
    Paused, gone or not-yet-ready clients never hold it up, and a lone client
    is aggregated the moment it has submitted.
    """
    if sess.status != "active" or rnd is None:
        return
    if rnd.state == "aggregating":
        since = rnd.aggregating_since or rnd.start_time
        if (utcnow() - since).total_seconds() > AGGREGATING_STALE_SECONDS:
            Round.query.filter_by(id=rnd.id, state="aggregating").update(
                {"state": "collecting", "aggregating_since": None}, synchronize_session=False
            )
            event(sess.session_id, None, "aggregation_recovered", f"round {rnd.round_number}")
            db.session.commit()
            db.session.refresh(rnd)
        else:
            return
    if rnd.state not in ("open", "collecting"):
        return

    cfg = sess.config()
    now = utcnow()
    updates = ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rnd.round_number).all()
    submitted = {u.client_id for u in updates}
    pending = _pending_live(rnd, submitted)
    active_pending = [c for c in pending if c.status == "active"]
    paused_pending = [c for c in pending if c.status == "paused"]
    timed_out = now >= rnd.deadline
    required = rnd.min_clients or 1

    # Nobody is working on this round and nothing has arrived: hand the round to every
    # online, active client (e.g. the only client was paused and has resumed, or a
    # newcomer arrived).  Nobody has to wait for them, so this is always safe.
    if not active_pending and not submitted:
        chosen = set(rnd.selected_list())
        newcomers = [c.client_id for c in live_clients(sess.session_id) if c.status == "active" and c.client_id not in chosen]
        if newcomers:
            Round.query.filter_by(id=rnd.id).update(
                {
                    "selected_clients": ",".join(rnd.selected_list() + newcomers),
                    "deadline": now + timedelta(seconds=cfg["round_timeout_seconds"]),
                },
                synchronize_session=False,
            )
            event(sess.session_id, None, "round_joined", f"round {rnd.round_number}: +{len(newcomers)} client(s)")
            db.session.commit()
            db.session.refresh(rnd)
            return

    if active_pending and not timed_out:
        return                                   # somebody is still training -> wait for them, and only them
    if len(submitted) >= required:
        _aggregate(sess, rnd, updates, dropped=[c for c in rnd.selected_list() if c not in submitted])
        return

    # Nothing to aggregate.  If everybody selected is merely paused (or the round was opened empty)
    # keep the session alive instead of failing the round.
    if (paused_pending or not rnd.selected_list()) and live_clients(sess.session_id):
        if (rnd.deadline - now).total_seconds() < cfg["round_timeout_seconds"] / 2:
            Round.query.filter_by(id=rnd.id).update(
                {"deadline": now + timedelta(seconds=cfg["round_timeout_seconds"])}, synchronize_session=False
            )
            db.session.commit()
            db.session.refresh(rnd)
        return
    _fail_round(sess, rnd, len(submitted), required, timed_out)


def _fail_round(sess, rnd, received, required, timed_out):
    """Participation threshold not met (SDS: re-open the round, abort after N attempts)."""
    why = "round timeout" if timed_out else "selected clients left"
    ok = Round.query.filter(Round.id == rnd.id, Round.state.in_(("open", "collecting"))).update(
        {"state": "failed", "end_time": utcnow()}, synchronize_session=False
    )
    if ok != 1:
        db.session.commit()
        return
    ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rnd.round_number).delete()
    event(sess.session_id, None, "round_failed", f"round {rnd.round_number}: {received}/{required} updates ({why})")
    attempt = (rnd.attempt or 1) + 1
    if attempt > MAX_ROUND_ATTEMPTS:
        _close_session(sess, "aborted", f"round {rnd.round_number} failed {MAX_ROUND_ATTEMPTS} times")
    else:
        _open_round(sess, rnd.round_number, attempt)
    db.session.commit()


def _aggregate(sess, rnd, updates: List[ClientUpdate], dropped: List[str]):
    claimed = Round.query.filter(Round.id == rnd.id, Round.state.in_(("open", "collecting"))).update(
        {"state": "aggregating", "aggregating_since": utcnow()}, synchronize_session=False
    )
    db.session.commit()
    if claimed != 1:
        return  # another request is already aggregating this round

    cfg = sess.config()
    t0 = time.perf_counter()
    try:
        weights = {u.client_id: weight_codec.unpack_blob(u.weights_blob) for u in updates}
        sizes = {u.client_id: int(u.dataset_size) for u in updates}
        agg = fedavg_aggregate(weights, sizes)
        agg = [np.asarray(a, dtype=np.float32) for a in agg]
        if not all(np.all(np.isfinite(a)) for a in agg):
            raise ValueError("aggregated weights contain NaN/Inf")
        agg_ms = (time.perf_counter() - t0) * 1000.0
        val_x, val_y = data_service.validation_set()
        ev = model_eval.evaluate(agg, val_x, val_y)
        # every client's OWN model of this round, per language (its own Per-Language chart)
        for u in updates:
            try:
                own = model_eval.evaluate(weights[u.client_id], val_x, val_y)
                u.val_accuracy = own["accuracy"]
                u.per_class_json = json.dumps(own["per_class"])
            except Exception:  # pragma: no cover - never block the round for a display value
                pass
    except Exception as exc:  # pragma: no cover - defensive
        Round.query.filter_by(id=rnd.id).update({"state": "failed", "end_time": utcnow()})
        _close_session(sess, "aborted", f"aggregation error: {exc}")
        db.session.commit()
        print(f"[AGGREGATE] round {rnd.round_number} FAILED: {exc}")
        return

    now = utcnow()
    n = rnd.round_number
    names = {c.client_id: c for c in Client.query.filter(Client.client_id.in_([u.client_id for u in updates])).all()}
    client_metrics = []
    for u in updates:
        c = names.get(u.client_id)
        client_metrics.append(
            {
                "client_id": u.client_id,
                "client_name": c.client_name if c else u.client_id[:8],
                "language": c.language if c else None,
                "backend": u.backend,
                "dataset_size": u.dataset_size,
                "local_accuracy": u.local_accuracy,
                "local_loss": u.local_loss,
                "train_seconds": u.train_seconds,
                "payload_bytes": u.payload_bytes,
            }
        )
    accs = [u.local_accuracy for u in updates if u.local_accuracy is not None]
    losses = [u.local_loss for u in updates if u.local_loss is not None]
    duration = (now - rnd.start_time).total_seconds()

    db.session.add(
        ModelCheckpoint(
            session_id=sess.session_id,
            round_number=n,
            weights_blob=weight_codec.pack_blob(agg),
            accuracy=ev["accuracy"],
            loss=ev["loss"],
            num_clients=len(updates),
        )
    )
    db.session.add(
        ExperimentLog(
            session_id=sess.session_id,
            round_number=n,
            accuracy=ev["accuracy"],
            loss=ev["loss"],
            precision_macro=ev["precision"],
            recall_macro=ev["recall"],
            f1_macro=ev["f1"],
            avg_local_accuracy=float(np.mean(accs)) if accs else None,
            avg_local_loss=float(np.mean(losses)) if losses else None,
            participating_clients=",".join(m["client_name"] for m in client_metrics),
            client_count=len(updates),
            payload_bytes=int(sum(u.payload_bytes or 0 for u in updates)),
            aggregation_ms=round(agg_ms, 2),
            round_duration_seconds=round(duration, 3),
            aggregation_method=cfg["aggregation_method"],
            per_class_json=json.dumps(ev["per_class"]),
            client_metrics_json=json.dumps(client_metrics),
            logged_at=now,
        )
    )
    Round.query.filter_by(id=rnd.id).update(
        {
            "state": "complete",
            "end_time": now,
            "duration_seconds": duration,
            "global_accuracy": ev["accuracy"],
            "global_loss": ev["loss"],
        }
    )
    for cid in dropped:
        event(sess.session_id, cid, "client_dropped", f"round {n}: no update before aggregation")
    for u in updates:
        u.weights_blob = None  # free space; only the aggregate is kept
    sess.model_version = n
    event(sess.session_id, None, "round_complete", f"round {n} acc={ev['accuracy']:.4f} clients={len(updates)}")

    # keep only the newest checkpoints with weights
    old = (
        ModelCheckpoint.query.filter(ModelCheckpoint.session_id == sess.session_id, ModelCheckpoint.round_number < n - (KEEP_CHECKPOINTS_WITH_WEIGHTS - 1))
        .all()
    )
    for ck in old:
        ck.weights_blob = None

    db.session.flush()
    if not _anyone_wants_more_rounds(sess):
        _close_session(sess, "completed", f"all {n} rounds finished (every client got the rounds it asked for)")
    else:
        _open_round(sess, n + 1)
    db.session.commit()
    print(f"[AGGREGATE] session {sess.session_id[:8]} round {n}: acc={ev['accuracy']:.3f} loss={ev['loss']:.3f} clients={len(updates)} ({agg_ms:.0f} ms)")


# ── client-facing views ──────────────────────────────────────────────────

def _latest_log(session_id: str) -> Optional[ExperimentLog]:
    return (
        ExperimentLog.query.filter_by(session_id=session_id).order_by(ExperimentLog.round_number.desc()).first()
    )


def current_round_view(client: Client) -> dict:
    sess = get_session(client.session_id)
    if sess is None:
        raise not_found("Session no longer exists", "session_not_found")
    touch(client)
    if not client.ready:                 # the browser has loaded the model + data and started polling
        client.ready = True
        db.session.commit()
    _maybe_start_session(sess)
    db.session.refresh(sess)
    rnd = latest_round(sess.session_id, sess.current_round) if sess.current_round else None
    if rnd is not None and sess.status == "active":
        process_round(sess, rnd)
        db.session.refresh(sess)
        rnd = latest_round(sess.session_id, sess.current_round) if sess.current_round else None

    cfg = sess.config()
    live = live_clients(sess.session_id)
    last = _latest_log(sess.session_id)
    mine_done = rounds_done_map(sess.session_id).get(client.client_id, 0)
    view = {
        "session_id": sess.session_id,
        "session_status": sess.status,
        "total_rounds": planned_rounds(sess),
        "my_rounds_wanted": _wanted(client, cfg),
        "my_rounds_done": mine_done,
        "round_number": sess.current_round,
        "round_state": rnd.state if rnd else "waiting",
        "rounds_completed": ExperimentLog.query.filter_by(session_id=sess.session_id).count(),
        "model_version": sess.model_version,
        "required_model_version": max(sess.current_round - 1, 0),
        "live_clients": len(live),
        "clients_total": Client.query.filter_by(session_id=sess.session_id).count(),
        "deadline": iso(rnd.deadline) if rnd else None,
        "server_time": iso(utcnow()),
        "client_status": client.status,
        "selected": False,
        "submitted": False,
        "received": 0,
        "phase": "waiting",
        "last_round": last.to_dict() if last else None,
    }
    if sess.status == "completed":
        view["phase"] = "session_complete"
    elif sess.status == "aborted":
        view["phase"] = "aborted"
    elif mine_done >= _wanted(client, cfg):
        view["phase"] = "my_rounds_done"        # this client got all the rounds it asked for; others carry on
    elif rnd is not None:
        submitted_ids = {
            u.client_id
            for u in ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rnd.round_number).all()
        }
        view["received"] = len(submitted_ids)
        view["selected"] = client.client_id in rnd.selected_list()
        view["submitted"] = client.client_id in submitted_ids
        view["selected_count"] = len(rnd.selected_list())
        if rnd.state in ("open", "collecting") and view["selected"] and not view["submitted"] and client.status == "active":
            view["phase"] = "train"
        elif view["submitted"] or (view["selected"] and rnd.state == "aggregating"):
            view["phase"] = "wait"
        else:
            view["phase"] = "not_selected"      # joined while this round was running -> starts with the next one
    return view


def client_shard_payload(client: Client) -> dict:
    sess = get_session(client.session_id)
    cfg = sess.config()
    pt, alpha = _client_part(client, cfg)
    x, y = data_service.client_shard(pt, client.partition_index or 0, client.language or "", alpha, cfg["seed"])
    import base64

    return {
        "num_samples": int(len(y)),
        "seq_len": int(x.shape[1]),
        "x_b64": base64.b64encode(np.ascontiguousarray(x, dtype="<u2").tobytes()).decode("ascii"),
        "labels": [int(v) for v in y],
        "label_distribution": data_service.shard_label_distribution(y),
        "language": client.language,
        "partition_type": pt,
        "partition_index": client.partition_index,
    }


def global_model_payload(session_id: str, since: Optional[int] = None) -> dict:
    sess = get_session(session_id)
    if sess is None:
        raise not_found("Session not found", "session_not_found")
    ck = (
        ModelCheckpoint.query.filter(ModelCheckpoint.session_id == session_id, ModelCheckpoint.weights_blob.isnot(None))
        .order_by(ModelCheckpoint.round_number.desc())
        .first()
    )
    if ck is None:
        raise not_found("No checkpoint stored for this session", "checkpoint_not_found")
    base = {
        "session_id": session_id,
        "version": ck.round_number,
        "round": ck.round_number,
        "architecture": model_spec.architecture(),
        "accuracy": ck.accuracy,
        "loss": ck.loss,
    }
    if since is not None and int(since) == ck.round_number:
        base["up_to_date"] = True
        base["weights"] = None
        return base
    base["up_to_date"] = False
    base["weights"] = weight_codec.encode_weights(weight_codec.unpack_blob(ck.weights_blob))
    return base


ALLOWED_UPDATE_KEYS = {
    "weights", "round", "round_number", "roundNumber", "client_id", "clientId", "dataset_size", "datasetSize",
    "local_epochs", "localEpochs", "backend_used", "backendUsed", "payload_size_bytes", "payloadSizeBytes",
    "metrics", "train_seconds", "trainSeconds", "model_id", "modelId",
}
ALLOWED_METRIC_KEYS = {"accuracy", "loss", "val_accuracy", "val_loss", "train_accuracy", "train_loss", "f1", "precision", "recall"}


def _metric(metrics, *keys):
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
            return float(v)
    return None


def submit_update(client: Client, data: dict, payload_bytes: int = 0) -> dict:
    """Receive one client's weights for the current round (US-10)."""
    if not isinstance(data, dict):
        raise bad_request("JSON object body required")
    # Privacy Enforcement Layer (server side, SDS 3.1.2.7): only model weights,
    # round identifiers and numeric metrics may be received - never raw text.
    unexpected = set(data.keys()) - ALLOWED_UPDATE_KEYS
    if unexpected:
        raise bad_request(
            f"Unexpected fields: {sorted(unexpected)}. Only weight tensors, round identifiers and numeric metrics are permitted.",
            sorted(unexpected)[0],
            "privacy_violation",
        )
    metrics = data.get("metrics") or {}
    if not isinstance(metrics, dict) or set(metrics.keys()) - ALLOWED_METRIC_KEYS:
        raise bad_request("'metrics' may only contain numeric accuracy/loss values", "metrics", "privacy_violation")

    sess = get_session(client.session_id)
    if sess is None or sess.status != "active":
        raise conflict("Session is not active", "session_not_active")
    rn = data.get("round") or data.get("round_number") or data.get("roundNumber")
    try:
        rn = int(rn)
    except (TypeError, ValueError):
        raise bad_request("'round' is required", "round")
    if rn != sess.current_round:
        raise conflict(f"Round {rn} is not the current round ({sess.current_round}).", "stale_round")
    rnd = latest_round(sess.session_id, rn)
    if rnd is None or rnd.state not in ("open", "collecting"):
        raise conflict("Round is no longer accepting updates", "round_closed")
    if client.client_id not in rnd.selected_list():
        raise forbidden("This client was not selected for the current round", "not_selected")
    if ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rn, client_id=client.client_id).first():
        raise conflict("An update for this round was already received", "duplicate_update")

    try:
        weights = weight_codec.decode_weights(data.get("weights"))
    except weight_codec.PayloadError as e:
        event(sess.session_id, client.client_id, "update_rejected", e.message)
        db.session.commit()
        raise bad_request(e.message, e.field, "invalid_weights")

    cfg = sess.config()
    pt, alpha = _client_part(client, cfg)
    x, y = data_service.client_shard(pt, client.partition_index or 0, client.language or "", alpha, cfg["seed"])
    upd = ClientUpdate(
        session_id=sess.session_id,
        round_number=rn,
        client_id=client.client_id,
        dataset_size=int(len(y)),          # server-derived: a client cannot inflate its own weight
        weights_blob=weight_codec.pack_blob(weights),
        local_accuracy=_metric(metrics, "accuracy", "train_accuracy", "val_accuracy"),
        local_loss=_metric(metrics, "loss", "train_loss", "val_loss"),
        train_seconds=_safe_float(data.get("train_seconds", data.get("trainSeconds")), 0, 36000),
        payload_bytes=int(payload_bytes or 0),
        backend=client.backend,
        local_epochs=_safe_int(data.get("local_epochs", data.get("localEpochs")), 1, 50),
    )
    db.session.add(upd)
    try:
        Round.query.filter(Round.id == rnd.id, Round.state == "open").update(
            {"state": "collecting"}, synchronize_session=False
        )
        client.last_seen = utcnow()
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise conflict("An update for this round was already received", "duplicate_update")

    db.session.refresh(sess)
    rnd = latest_round(sess.session_id, rn)
    process_round(sess, rnd)
    db.session.refresh(sess)
    rnd = latest_round(sess.session_id, rn)
    received = ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rn).count()
    return {
        "status": "accepted",
        "message": "Update received",
        "round": rn,
        "round_state": rnd.state,
        "aggregated": rnd.state == "complete",
        "updates_received": received,
        "required": rnd.min_clients,
        "selected": len(rnd.selected_list()),
        "model_version": sess.model_version,
    }


# ── dashboard / metrics ──────────────────────────────────────────────────

def session_overview(sess: Optional[ExperimentSession], client_id: Optional[str] = None) -> dict:
    if sess is None:
        return {"session": None, "summary": None, "clients": [], "rounds": [], "model": _model_info()}
    cfg = sess.config()
    live_ids = {c.client_id for c in live_clients(sess.session_id)}
    rnd = latest_round(sess.session_id, sess.current_round) if sess.current_round else None
    selected = set(rnd.selected_list()) if rnd else set()
    submitted = set()
    if rnd:
        submitted = {
            u.client_id
            for u in ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rnd.round_number).all()
        }
    clients = []
    done_map = rounds_done_map(sess.session_id)
    rows = Client.query.filter_by(session_id=sess.session_id).order_by(Client.partition_index).all()
    for c in rows:
        online = c.client_id in live_ids
        clients.append(
            {
                **c.to_dict(),
                "rounds_done": done_map.get(c.client_id, 0),
                "rounds_wanted": _wanted(c, cfg),
                "online": online,
                "selected_this_round": c.client_id in selected,
                "submitted_this_round": c.client_id in submitted,
            }
        )
    logs = ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number).all()
    latest = logs[-1] if logs else None
    contributed = db.session.query(ClientUpdate.client_id).filter_by(session_id=sess.session_id).distinct().count()
    # GLOBAL numbers: the aggregated model of ALL clients (any device), evaluated by the server.
    summary = {
        "status": sess.status,
        "rounds_completed": len(logs),
        # GLOBAL "rounds": every client-round that was averaged into the global result
        # (client 1: 10 + client 2: 7 + client 3: 5 = 22).  rounds_completed above is the
        # number of aggregation steps of the shared model.
        "client_rounds": int(sum(l.client_count or 0 for l in logs)),
        "client_rounds_planned": int(sum(
            max(done_map.get(c.client_id, 0), _wanted(c, cfg) if c.status != "left" else 0) for c in rows
        )),
        "aggregation_rounds": len(logs),
        "total_rounds": planned_rounds(sess, rows, done_map),
        "current_round": sess.current_round,
        "accuracy": latest.accuracy if latest else None,
        "loss": latest.loss if latest else None,
        "f1": latest.f1_macro if latest else None,
        "precision": latest.precision_macro if latest else None,
        "recall": latest.recall_macro if latest else None,
        "clients_total": len(clients),
        "clients_online": sum(1 for c in clients if c["online"]),
        "clients_contributed": contributed,
        "clients_last_round": latest.client_count if latest else 0,
        "devices": sorted({c["backend"] for c in clients if c.get("backend")}),
    }
    return {
        "session": sess.to_dict(),
        "summary": summary,
        "current_round": (rnd.to_dict() if rnd else None),
        "received": len(submitted),
        "clients": clients,
        "rounds": [l.to_dict() for l in logs],
        "latest": latest.to_dict() if latest else None,
        "my": _my_results(sess, client_id, cfg, done_map, logs) if client_id else None,
        "partition_indicator": "non_iid" if any(_client_part(c, cfg)[0] != "iid" for c in rows) else "iid",
        "model": _model_info(),
        "server_time": iso(utcnow()),
    }


def _my_results(sess, client_id, cfg, done_map, logs) -> Optional[dict]:
    """Everything that belongs to ONE client: its own round history, graphs and summary."""
    c = Client.query.filter_by(client_id=client_id, session_id=sess.session_id).first()
    if c is None:
        return None
    gl = {l.round_number: l for l in logs}
    ups = (
        ClientUpdate.query.filter_by(session_id=sess.session_id, client_id=client_id)
        .order_by(ClientUpdate.round_number).all()
    )
    rounds = []
    for i, u in enumerate(ups, start=1):
        g = gl.get(u.round_number)
        rounds.append({
            "my_round": i,
            "round": u.round_number,
            "local_accuracy": u.local_accuracy,
            "local_loss": u.local_loss,
            "train_seconds": u.train_seconds,
            "dataset_size": u.dataset_size,
            "backend": u.backend,
            "val_accuracy": u.val_accuracy,
            "per_class": json.loads(u.per_class_json) if u.per_class_json else None,
            "global_accuracy": g.accuracy if g else None,
            "global_loss": g.loss if g else None,
            "global_f1": g.f1_macro if g else None,
            "timestamp": iso(u.created_at),
        })
    accs = [r["local_accuracy"] for r in rounds if r["local_accuracy"] is not None]
    losses = [r["local_loss"] for r in rounds if r["local_loss"] is not None]
    secs = [r["train_seconds"] for r in rounds if r["train_seconds"] is not None]
    pt, alpha = _client_part(c, cfg)
    return {
        "client": {**c.to_dict(), "partition_type": pt, "dirichlet_alpha": alpha},
        "rounds": rounds,
        "summary": {
            "rounds_done": done_map.get(c.client_id, 0),
            "rounds_wanted": _wanted(c, cfg),
            "joined_round": c.joined_round or 0,
            "samples": rounds[-1]["dataset_size"] if rounds else None,
            "last_local_accuracy": accs[-1] if accs else None,
            "best_local_accuracy": max(accs) if accs else None,
            "avg_local_accuracy": float(np.mean(accs)) if accs else None,
            "last_local_loss": losses[-1] if losses else None,
            "avg_train_seconds": float(np.mean(secs)) if secs else None,
            "total_train_seconds": float(np.sum(secs)) if secs else 0.0,
            "global_accuracy_at_last_round": rounds[-1]["global_accuracy"] if rounds else None,
        },
    }


def _model_info():
    return {
        "name": model_spec.ARCHITECTURE_NAME,
        "parameters": model_spec.total_parameters(),
        "size_bytes": model_spec.model_size_bytes(),
        "size_mb": round(model_spec.model_size_bytes() / 1e6, 3),
        "limit_mb": 5,
        "layers": [
            "Embedding(5000x32)", "GlobalAveragePooling1D", "Dense(16, relu)", "Dropout(0.3)", "Dense(3, softmax)"
        ],
    }
