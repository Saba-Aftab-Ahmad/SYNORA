-- ============================================================================
-- Synora v2 migration (PostgreSQL / Supabase)
-- ----------------------------------------------------------------------------
-- OPTIONAL: the server runs the same migration automatically at start-up
-- (server/database.py -> ensure_schema).  Use this file only if you prefer to
-- apply it by hand in the Supabase SQL editor.  Safe to run more than once.
-- ============================================================================

-- 1. new columns on existing tables (all nullable) ---------------------------
ALTER TABLE clients ADD COLUMN IF NOT EXISTS session_id        VARCHAR(36);
ALTER TABLE clients ADD COLUMN IF NOT EXISTS partition_index   INTEGER;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS language          VARCHAR(20);
ALTER TABLE clients ADD COLUMN IF NOT EXISTS session_token     VARCHAR(64);
ALTER TABLE clients ADD COLUMN IF NOT EXISTS cpu_cores         INTEGER;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS memory_gb         DOUBLE PRECISION;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS backend           VARCHAR(20);
ALTER TABLE clients ADD COLUMN IF NOT EXISTS webgpu_available  BOOLEAN;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS webnn_available   BOOLEAN;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS capability_score  DOUBLE PRECISION;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS user_agent        VARCHAR(200);
ALTER TABLE clients ADD COLUMN IF NOT EXISTS last_seen         TIMESTAMP;

ALTER TABLE rounds ADD COLUMN IF NOT EXISTS session_id         VARCHAR(36);
ALTER TABLE rounds ADD COLUMN IF NOT EXISTS selected_clients   TEXT;
ALTER TABLE rounds ADD COLUMN IF NOT EXISTS deadline           TIMESTAMP;
ALTER TABLE rounds ADD COLUMN IF NOT EXISTS attempt            INTEGER;
ALTER TABLE rounds ADD COLUMN IF NOT EXISTS aggregating_since  TIMESTAMP;

ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS session_id             VARCHAR(36);
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS precision_macro        DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS recall_macro           DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS f1_macro               DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS avg_local_accuracy     DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS avg_local_loss         DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS payload_bytes          INTEGER;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS aggregation_ms         DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS round_duration_seconds DOUBLE PRECISION;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS aggregation_method     VARCHAR(30);
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS per_class_json         TEXT;
ALTER TABLE experiment_logs ADD COLUMN IF NOT EXISTS client_metrics_json    TEXT;

-- 2. new tables --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_sessions (
    session_id    VARCHAR(36) PRIMARY KEY,
    status        VARCHAR(20),
    config_json   TEXT NOT NULL,
    current_round INTEGER,
    model_version INTEGER,
    created_at    TIMESTAMP,
    started_at    TIMESTAMP,
    ended_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS client_updates (
    id             SERIAL PRIMARY KEY,
    session_id     VARCHAR(36) NOT NULL,
    round_number   INTEGER NOT NULL,
    client_id      VARCHAR(36) NOT NULL,
    dataset_size   INTEGER NOT NULL,
    weights_blob   TEXT,
    local_accuracy DOUBLE PRECISION,
    local_loss     DOUBLE PRECISION,
    train_seconds  DOUBLE PRECISION,
    payload_bytes  INTEGER,
    backend        VARCHAR(20),
    local_epochs   INTEGER,
    created_at     TIMESTAMP,
    CONSTRAINT uq_update_session_round_client UNIQUE (session_id, round_number, client_id)
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id           SERIAL PRIMARY KEY,
    session_id   VARCHAR(36) NOT NULL,
    round_number INTEGER NOT NULL,
    weights_blob TEXT,
    accuracy     DOUBLE PRECISION,
    loss         DOUBLE PRECISION,
    num_clients  INTEGER,
    created_at   TIMESTAMP,
    CONSTRAINT uq_checkpoint_session_round UNIQUE (session_id, round_number)
);

CREATE TABLE IF NOT EXISTS client_selection_logs (
    id               SERIAL PRIMARY KEY,
    session_id       VARCHAR(36) NOT NULL,
    round_number     INTEGER NOT NULL,
    client_id        VARCHAR(36) NOT NULL,
    client_name      VARCHAR(100),
    strategy         VARCHAR(20),
    selected         BOOLEAN,
    capability_score DOUBLE PRECISION,
    reason           VARCHAR(200),
    created_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS client_events (
    id         SERIAL PRIMARY KEY,
    session_id VARCHAR(36),
    client_id  VARCHAR(36),
    event_type VARCHAR(30) NOT NULL,
    detail     VARCHAR(300),
    created_at TIMESTAMP
);

-- 3. indexes -----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_clients_session_id           ON clients (session_id);
CREATE INDEX IF NOT EXISTS ix_rounds_session_id            ON rounds (session_id);
CREATE INDEX IF NOT EXISTS ix_experiment_logs_session_id   ON experiment_logs (session_id);
CREATE INDEX IF NOT EXISTS ix_client_updates_session_id    ON client_updates (session_id);
CREATE INDEX IF NOT EXISTS ix_model_checkpoints_session_id ON model_checkpoints (session_id);
CREATE INDEX IF NOT EXISTS ix_client_selection_logs_session_id ON client_selection_logs (session_id);
CREATE INDEX IF NOT EXISTS ix_client_events_session_id     ON client_events (session_id);

-- 4. SECURITY (recommended on Supabase) --------------------------------------
-- The Flask server connects with the database owner role, which bypasses RLS.
-- Enabling RLS (with no policies) blocks anonymous access through Supabase's
-- public REST API while the Flask server keeps working.
ALTER TABLE clients                ENABLE ROW LEVEL SECURITY;
ALTER TABLE rounds                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiment_config      ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiment_logs        ENABLE ROW LEVEL SECURITY;
ALTER TABLE aggregation_config     ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiment_sessions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_updates         ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_checkpoints      ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_selection_logs  ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_events          ENABLE ROW LEVEL SECURITY;
