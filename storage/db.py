import os
import json
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv
import numpy as np

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   os.getenv("PG_DB",       "chexpert"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

# ------------------------------------------------------------------ #
# Connection                                                           #
# ------------------------------------------------------------------ #

@contextmanager
def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur


# ------------------------------------------------------------------ #
# pipeline_runs                                                        #
# ------------------------------------------------------------------ #

def start_pipeline_run() -> int:
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_runs (status) VALUES ('running') RETURNING id"
        )
        return cur.fetchone()["id"]


def finish_pipeline_run(run_id: int, studies_in: int, studies_out: int, status="complete"):
    with get_cursor() as cur:
        cur.execute(
            """UPDATE pipeline_runs
                SET status = %s, studies_in = %s, studies_out = %s
                WHERE id = %s""",
            (status, studies_in, studies_out, run_id),
        )


# ------------------------------------------------------------------ #
# patients                                                             #
# ------------------------------------------------------------------ #

def upsert_patient(subject_id: str):
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO patients (subject_id) VALUES (%s) ON CONFLICT (subject_id) DO NOTHING",
            (subject_id,),
        )


# ------------------------------------------------------------------ #
# studies                                                              #
# ------------------------------------------------------------------ #

def upsert_study(study_id: str, subject_id: str, split: str,
                run_id: int, report_txt: str = None):
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO studies (study_id, subject_id, split, run_id, report_txt)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (study_id) DO UPDATE
                SET run_id     = EXCLUDED.run_id,
                    report_txt = COALESCE(EXCLUDED.report_txt, studies.report_txt)""",
            (study_id, subject_id, split, run_id, report_txt),
        )


# ------------------------------------------------------------------ #
# label_mappings                                                       #
# ------------------------------------------------------------------ #

# U-Zeroes for these two; U-Ones for everything else
UZERO_CONDITIONS = {"Cardiomegaly", "Atelectasis"}


def map_label(condition: str, raw_value: str) -> tuple[int, str]:
    """
    Apply U-Ones / U-Zeroes strategy. Returns (mapped_int, strategy_name).

    raw_value meanings from CheXpert CSV:
        '1.0'  → confirmed positive
        '0.0'  → confirmed negative
        '-1.0' → uncertain (radiologist hedged)
        ''     → not mentioned — treat as negative
    """
    if raw_value == "1.0":
        return 1, "positive"
    if raw_value in ("0.0", ""):
        return 0, "negative"
    if raw_value == "-1.0":
        if condition in UZERO_CONDITIONS:
            return 0, "U-Zeroes"
        return 1, "U-Ones"
    return 0, "negative"  # fallback for unexpected values


def insert_labels(study_id: str, raw_labels: dict[str, str]):
    rows = []
    for condition, raw_value in raw_labels.items():
        mapped, strategy = map_label(condition, raw_value)
        rows.append((study_id, condition, raw_value, mapped, strategy))

    with get_cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO label_mappings (study_id, condition, raw_value, mapped_value, strategy)
                VALUES %s
                ON CONFLICT (study_id, condition) DO NOTHING""",
            rows,
        )


# ------------------------------------------------------------------ #
# embeddings                                                           #
# ------------------------------------------------------------------ #

def insert_embedding(study_id: str, dicom_uid: str,
                    faiss_id: int, model_name: str, embed_dim: int):
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO embeddings (study_id, dicom_uid, faiss_id, model_name, embed_dim)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dicom_uid) DO NOTHING""",
            (study_id, dicom_uid, faiss_id, model_name, embed_dim),
        )


def mark_dicom_deleted(dicom_uid: str):
    with get_cursor() as cur:
        cur.execute(
            "UPDATE embeddings SET dicom_deleted = TRUE WHERE dicom_uid = %s",
            (dicom_uid,),
        )


def get_faiss_id(dicom_uid: str) -> int | None:
    with get_cursor() as cur:
        cur.execute("SELECT faiss_id FROM embeddings WHERE dicom_uid = %s", (dicom_uid,))
        row = cur.fetchone()
        return row["faiss_id"] if row else None


def get_study_id_by_faiss(faiss_id: int) -> str | None:
    """Reverse lookup: faiss_id → study_id. Used after retrieval."""
    with get_cursor() as cur:
        cur.execute("SELECT study_id FROM embeddings WHERE faiss_id = %s", (faiss_id,))
        row = cur.fetchone()
        return row["study_id"] if row else None


# ------------------------------------------------------------------ #
# outputs                                                              #
# ------------------------------------------------------------------ #

def insert_output(study_id: str, output_dict: dict,
                    confidence: float, latency_ms: int, heatmap_path: str = None):
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO outputs (study_id, output_json, confidence, latency_ms, heatmap_path)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (study_id) DO UPDATE
                SET output_json  = EXCLUDED.output_json,
                    confidence   = EXCLUDED.confidence,
                    latency_ms   = EXCLUDED.latency_ms,
                    heatmap_path = EXCLUDED.heatmap_path""",
            (study_id, json.dumps(output_dict), confidence, latency_ms, heatmap_path),
        )


def mark_output_validated(study_id: str):
    with get_cursor() as cur:
        cur.execute(
            "UPDATE outputs SET validated = TRUE WHERE study_id = %s", (study_id,)
        )


def insert_failed_encoding(study_id: str, dicom_path: str, error: str):
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO failed_encodings (study_id, dicom_path, error)
                VALUES (%s, %s, %s)""",
            (study_id, dicom_path, error),
        )

def insert_embedding(study_id: str, dicom_uid: str,
                     faiss_id: int, model_name: str, embed_dim: int,
                     feature_map: bytes = None, feature_map_shape: list = None):
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO embeddings
                   (study_id, dicom_uid, faiss_id, model_name, embed_dim,
                    feature_map, feature_map_shape)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (dicom_uid) DO NOTHING""",
            (study_id, dicom_uid, faiss_id, model_name, embed_dim,
             psycopg2.Binary(feature_map) if feature_map else None,
             feature_map_shape),
        )

# ------------------------------------------------------------------ #
# reports                                                              #
# ------------------------------------------------------------------ #

def insert_report(
    study_id: str,
    findings: str,
    impression: str,
    full_report: str,
    report_embedding: "np.ndarray",
):
    """
    Persist the BioViL-T generated report and its 128-dim text embedding.
    report_embedding is stored as raw bytes (float32, C-contiguous).
    """
    import numpy as np
    embedding_bytes = np.ascontiguousarray(report_embedding).tobytes()

    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO reports
                    (study_id, findings, impression, full_report, report_embedding)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (study_id) DO UPDATE
                   SET findings         = EXCLUDED.findings,
                       impression       = EXCLUDED.impression,
                       full_report      = EXCLUDED.full_report,
                       report_embedding = EXCLUDED.report_embedding""",
            (study_id, findings, impression, full_report,
             psycopg2.Binary(embedding_bytes)),
        )


def get_report_embedding(study_id: str) -> "np.ndarray | None":
    """
    Load the 128-dim report embedding for a study.
    Returns a float32 numpy array, or None if not yet generated.
    """
    import numpy as np
    with get_cursor() as cur:
        cur.execute(
            "SELECT report_embedding FROM reports WHERE study_id = %s",
            (study_id,),
        )
        row = cur.fetchone()
    if row is None or row["report_embedding"] is None:
        return None
    return np.frombuffer(bytes(row["report_embedding"]), dtype=np.float32).copy()


def insert_diagnosis(study_id: str, probabilities: dict[str, float]):
    """
    Insert per-condition probability scores from CheXNet.
    probabilities: {chexpert_condition: float between 0 and 1}
    """
    rows = [(study_id, condition, prob)
            for condition, prob in probabilities.items()]

    with get_cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO diagnoses (study_id, condition, probability)
                VALUES %s
                ON CONFLICT (study_id, condition) DO UPDATE
                SET probability = EXCLUDED.probability""",
            rows,
        )