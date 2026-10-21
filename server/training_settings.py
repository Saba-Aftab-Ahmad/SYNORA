"""
server/training_settings.py
===========================
THE one place where every training setting that is decided BY THE SERVER lives.

Nobody can change these from the browser: the training page does not show
them and the server ignores them if a client sends them.  Every client of a
session therefore trains with exactly the same values, which is what makes
the federated average meaningful.

How the values were chosen (numpy simulation of the exact Synora model on the
real 3-language data, 1 and 3 IID clients x 2000 samples, 10 rounds, 24
combinations of learning rate / local epochs / batch size, Adam optimiser):

  * learning rate 0.01  - the best accuracy at round 10 for both 1 and 3 clients
    (>= 0.02 over-fits after ~5 rounds: single client fell from 96 % to 90-94 %;
    0.005 is simply slower).
  * local epochs 2      - reaches ~95-96 % after 2-3 rounds, against 4-5 rounds
    for 1 epoch.  Every round also costs a fixed ~2-3 s (upload + download of the
    0.64 MB model), so fewer rounds is the faster way to the same accuracy.
  * batch size 64       - half the optimiser steps of batch 32 (about 2x faster
    on CPU/GPU) for practically the same final accuracy.

Selection, threshold and timeout are decided in code as well:

  * MIN_PARTICIPATION_THRESHOLD = 1: a round is aggregated as soon as ONE update
    has arrived from the clients that are really working on it.  A lone client
    never waits for anybody (SRS US-11 - the threshold stays a server-side
    parameter, it is just fixed to its minimum).
  * Client selection is AUTOMATIC (SRS US-12 / US-13): every ready client takes
    part; only when more than MAX_CLIENTS_PER_ROUND clients are online does the
    server choose - mostly by capability score (resource-aware, US-13) with a few
    random "fairness" places (random baseline, US-12) so weak devices still get a
    turn.
"""

# ── local training (sent to every client with the session) ───────────────
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 2
BATCH_SIZE = 64

# ── round control ────────────────────────────────────────────────────────
ROUND_TIMEOUT_SECONDS = 120          # SDS: 120 s local-training / round window
MIN_PARTICIPATION_THRESHOLD = 1      # SRS US-11 (server-side parameter)

# ── automatic client selection ───────────────────────────────────────────
MAX_CLIENTS_PER_ROUND = 10           # more online clients than this -> server selects
FAIRNESS_FRACTION = 0.25             # share of the places given by random draw
MAX_LIVE_CLIENTS = 30                # safety cap for the free-tier host (one constant to raise)

# ── misc ─────────────────────────────────────────────────────────────────
SEED = 42
AGGREGATION_METHOD = "fedavg"        # FedProx is future work in the SRS/SDS
SELECTION_STRATEGY = "auto"

# Settings the person who STARTS a training may choose (training page):
USER_DEFAULTS = {
    "total_rounds": 20,
    "partition_type": "iid",         # iid | language | dirichlet
    "dirichlet_alpha": 0.5,
}

# Settings that only the server decides:
FIXED_SETTINGS = {
    "learning_rate": LEARNING_RATE,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "round_timeout_seconds": ROUND_TIMEOUT_SECONDS,
    "min_participation_threshold": MIN_PARTICIPATION_THRESHOLD,
    "max_clients_per_round": MAX_CLIENTS_PER_ROUND,
    "selection_strategy": SELECTION_STRATEGY,
    "aggregation_method": AGGREGATION_METHOD,
    "seed": SEED,
}

DEFAULT_CONFIG = {**USER_DEFAULTS, **FIXED_SETTINGS}
