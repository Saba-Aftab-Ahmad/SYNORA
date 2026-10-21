# SYNORA
Browser-based federated learning toolkit for privacy-preserving text classification of low-resource Kenyan languages (Dholuo, Kalenjin, Kidawida). Local AI training runs entirely in the browser using TensorFlow.js,  raw data never leaves the device.


## Version 2 - what runs now

* **Task:** language identification (text classification) - Dholuo / Kalenjin / Kidawida. The Swahili column is not used; no translation takes place.
* **Model:** Embedding(5000x32) -> average pooling -> Dense(16) -> Dropout -> Dense(3); 160,579 parameters (0.64 MB, limit 5 MB). Defined once in `server/model_spec.py`; the browser builds the same model from the server's description. The server creates the initial global weights.
* **Coordination:** all state is stored in the database (`server/fl_service.py`) - sessions, rounds, updates, checkpoints, selection log - so several gunicorn workers, several browsers and server restarts are safe.
* **API:** `/api/...` (see `server/coordination_server.py`); old routes remain as wrappers.
* **Evaluation:** after each round the server evaluates the aggregated model on a held-out validation set (`server/model_eval.py`): accuracy, loss, macro precision/recall/F1, per-language results.

### Run locally (no real database touched)
```
python -m venv venv && venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                               # DATABASE_URL=sqlite:///local_test.db
python -m pytest tests/test_server_v2.py -q          # 45 tests
python server/coordination_server.py                 # http://127.0.0.1:5000
```
See `RUN_GUIDE.md` for the complete step-by-step guide.
