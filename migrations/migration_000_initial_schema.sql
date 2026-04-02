-- migration_000_initial_schema.sql
-- Run this FIRST, before any pipeline scripts or other migrations.
-- Tables are created in foreign-key dependency order.
-- schema.sql has been deleted; this file is the source of truth.

-- ------------------------------------------------------------------ --
-- pipeline_runs                                                        --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          SERIAL      PRIMARY KEY,
    status      TEXT        NOT NULL DEFAULT 'running',  -- running | complete | failed
    studies_in  INTEGER,
    studies_out INTEGER,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- ------------------------------------------------------------------ --
-- patients                                                             --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS patients (
    subject_id  TEXT        PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------ --
-- studies  (depends on: patients, pipeline_runs)                      --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS studies (
    study_id    TEXT        PRIMARY KEY,
    subject_id  TEXT        NOT NULL REFERENCES patients(subject_id),
    split       TEXT        CHECK (split IN ('train', 'val', 'test')),
    run_id      INTEGER     REFERENCES pipeline_runs(id),
    report_txt  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_studies_subject ON studies(subject_id);
CREATE INDEX IF NOT EXISTS idx_studies_split   ON studies(split);
CREATE INDEX IF NOT EXISTS idx_studies_run     ON studies(run_id);

-- ------------------------------------------------------------------ --
-- label_mappings  (depends on: studies)                               --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS label_mappings (
    id           SERIAL  PRIMARY KEY,
    study_id     TEXT    NOT NULL REFERENCES studies(study_id),
    condition    TEXT    NOT NULL,
    raw_value    TEXT,                                   -- '1.0' | '-1.0' | '0.0' | ''
    mapped_value INTEGER NOT NULL CHECK (mapped_value IN (0, 1)),
    strategy     TEXT    NOT NULL,                       -- positive | negative | U-Ones | U-Zeroes
    UNIQUE (study_id, condition)
);

CREATE INDEX IF NOT EXISTS idx_labels_study     ON label_mappings(study_id);
CREATE INDEX IF NOT EXISTS idx_labels_condition ON label_mappings(condition);

-- ------------------------------------------------------------------ --
-- embeddings  (depends on: studies)                                   --
-- float32 vectors live in FAISS; faiss_id is the bridge back here.   --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS embeddings (
    id            SERIAL      PRIMARY KEY,
    study_id      TEXT        NOT NULL REFERENCES studies(study_id),
    dicom_uid     TEXT        NOT NULL UNIQUE,
    faiss_id      BIGINT      NOT NULL UNIQUE,           -- BIGINT: FAISS indexes can exceed 2B entries
    model_name    TEXT        NOT NULL,
    embed_dim     INTEGER     NOT NULL,
    dicom_deleted BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_study ON embeddings(study_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON embeddings(faiss_id);

-- ------------------------------------------------------------------ --
-- reports  (depends on: studies)                                      --
-- Output of Model 2 (BioViL-T). report_embedding is 128-dim float32  --
-- stored as raw BYTEA; use np.frombuffer(..., dtype=np.float32).      --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS reports (
    study_id         TEXT        PRIMARY KEY REFERENCES studies(study_id),
    findings         TEXT,
    impression       TEXT,
    full_report      TEXT,
    report_embedding BYTEA,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------ --
-- diagnoses  (depends on: studies)                                    --
-- Output of Model 3 (CheXNet), one row per (study, condition).       --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS diagnoses (
    id          SERIAL  PRIMARY KEY,
    study_id    TEXT    NOT NULL REFERENCES studies(study_id),
    condition   TEXT    NOT NULL,
    probability FLOAT   NOT NULL CHECK (probability BETWEEN 0 AND 1),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (study_id, condition)
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_study     ON diagnoses(study_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_condition ON diagnoses(condition);

-- ------------------------------------------------------------------ --
-- outputs  (depends on: studies)                                      --
-- Final JSON per study after Model 4 validation.                      --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS outputs (
    study_id     TEXT    PRIMARY KEY REFERENCES studies(study_id),
    output_json  JSONB   NOT NULL,
    confidence   FLOAT   CHECK (confidence BETWEEN 0 AND 1),
    latency_ms   INTEGER,
    heatmap_path TEXT,
    validated    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);