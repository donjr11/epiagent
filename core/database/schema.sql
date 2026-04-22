-- EpiAgent v1 SQLite schema. Mirrors Section 8 of the design doc.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    id                INTEGER PRIMARY KEY,
    huggingface_id    TEXT UNIQUE NOT NULL,
    name              TEXT NOT NULL,
    provider          TEXT,
    parameter_count_b REAL,
    license_spdx      TEXT,
    is_on_whitelist   INTEGER NOT NULL DEFAULT 0,
    context_window    INTEGER,
    release_date      DATE,
    first_seen_at     TIMESTAMP NOT NULL,
    last_seen_at      TIMESTAMP NOT NULL,
    is_active         INTEGER NOT NULL DEFAULT 1,
    arena_elo         REAL
);

CREATE TABLE IF NOT EXISTS model_capabilities (
    model_id   INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    capability TEXT NOT NULL,
    PRIMARY KEY (model_id, capability)
);

-- Append-only; scorer reads MAX(measured_at) per tuple.
CREATE TABLE IF NOT EXISTS model_benchmarks (
    model_id     INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    metric_name  TEXT NOT NULL,
    source       TEXT NOT NULL,
    n_shot       INTEGER NOT NULL DEFAULT -1,
    cot          INTEGER NOT NULL DEFAULT 0,
    score        REAL NOT NULL,
    measured_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (model_id, metric_name, source, n_shot, cot, measured_at)
);

CREATE TABLE IF NOT EXISTS model_hardware_requirements (
    model_id            INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    quantization        TEXT NOT NULL,
    min_vram_gb         REAL NOT NULL,
    recommended_vram_gb REAL,
    min_ram_gb          REAL,
    estimator_source    TEXT,
    estimated_at        TIMESTAMP,
    PRIMARY KEY (model_id, quantization)
);

-- One row per capability; weights from BWM.
CREATE TABLE IF NOT EXISTS scoring_profiles (
    id                 INTEGER PRIMARY KEY,
    capability         TEXT UNIQUE NOT NULL,
    version            INTEGER NOT NULL,
    criteria           TEXT NOT NULL,           -- JSON list
    weights            TEXT NOT NULL,           -- JSON {metric: weight}
    source_preferences TEXT NOT NULL,           -- JSON per-metric ordered
    bwm_input          TEXT,                    -- JSON (best, worst, A_B, A_W)
    consistency_ratio  REAL,
    created_at         TIMESTAMP NOT NULL,
    CHECK (consistency_ratio IS NULL OR consistency_ratio <= 0.10)
);

-- Query history for auditability.
CREATE TABLE IF NOT EXISTS scoring_runs (
    id               INTEGER PRIMARY KEY,
    capabilities     TEXT NOT NULL,
    model_id         INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    rank             INTEGER NOT NULL,
    combined_score   REAL NOT NULL,
    per_cap_scores   TEXT NOT NULL,
    contributions    TEXT NOT NULL,
    user_vram_gb     REAL,
    user_context_req INTEGER,
    run_at           TIMESTAMP NOT NULL
);

-- Audit log for model-level metadata overwrites.
CREATE TABLE IF NOT EXISTS model_change_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id       INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    field          TEXT NOT NULL,
    old_value      TEXT,
    new_value      TEXT,
    changed_at     TIMESTAMP NOT NULL
);

-- Persist daily digest state so the notifier can detect changes.
CREATE TABLE IF NOT EXISTS notification_state (
    id              INTEGER PRIMARY KEY,
    last_digest_at  TIMESTAMP,
    last_snapshot   TEXT             -- JSON {capability: [(model_id, rank), ...]}
);

-- Ingestion run log.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          TEXT NOT NULL,        -- ok|partial|failed
    candidates_seen INTEGER,
    models_upserted INTEGER,
    warnings        TEXT                   -- JSON list
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_model_metric
    ON model_benchmarks(model_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_hw_model ON model_hardware_requirements(model_id);
CREATE INDEX IF NOT EXISTS idx_caps_capability ON model_capabilities(capability);
