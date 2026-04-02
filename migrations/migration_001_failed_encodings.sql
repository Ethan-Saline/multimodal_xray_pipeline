CREATE TABLE IF NOT EXISTS failed_encodings (
    id          SERIAL PRIMARY KEY,
    study_id    TEXT        NOT NULL REFERENCES studies(study_id),
    dicom_path  TEXT        NOT NULL,
    error       TEXT        NOT NULL,
    attempted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_failed_study ON failed_encodings(study_id);
