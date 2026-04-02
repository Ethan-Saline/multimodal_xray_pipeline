"""
diagnose.py — Diagnosis (Model 3)
pipeline/ | CheXpert Pipeline

For each study that has both an image embedding and a report embedding
but no entry in the diagnoses table yet:
  1. Load image embedding from FAISS
  2. Load report embedding from PostgreSQL
  3. Run MultimodalClassifier.predict()
  4. Write per-condition probability scores to the diagnoses table

Must run AFTER report.py and BEFORE validate.py.

Usage:
    python -m pipeline.diagnose
    python -m pipeline.diagnose --study-id 50414267
    python -m pipeline.diagnose --split val
    python -m pipeline.diagnose --limit 100
"""

import argparse
import time

import numpy as np

from models.classifier import load_model, predict, DEFAULT_WEIGHTS_PATH
from storage.db import get_cursor, insert_diagnosis
from storage.faiss_store import FAISSStore


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_pending_studies(study_id: str = None, split: str = None) -> list[dict]:
    """
    Return studies that have both an image embedding and a report embedding
    but no entry in the diagnoses table yet.
    """
    with get_cursor() as cur:
        query = """
            SELECT
                s.study_id,
                s.split,
                e.faiss_id,
                r.report_embedding
            FROM   studies    s
            INNER JOIN embeddings e ON e.study_id = s.study_id
            INNER JOIN reports    r ON r.study_id = s.study_id
            LEFT  JOIN diagnoses  d ON d.study_id = s.study_id
            WHERE  d.study_id        IS NULL
              AND  r.report_embedding IS NOT NULL
        """
        params = []

        if study_id:
            query += " AND s.study_id = %s"
            params.append(study_id)
        elif split:
            query += " AND s.split = %s"
            params.append(split)

        query += " ORDER BY s.study_id"
        cur.execute(query, params)
        # De-duplicate — one row per study (should already be unique but be safe)
        seen = set()
        rows = []
        for row in cur.fetchall():
            if row["study_id"] not in seen:
                seen.add(row["study_id"])
                rows.append(dict(row))
        return rows


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(study_id: str = None, split: str = None, limit: int = None):
    """
    For each pending study:
      1. Reconstruct image embedding from FAISS
      2. Deserialise report embedding from PostgreSQL BYTEA
      3. Run classifier inference
      4. Persist per-condition probabilities to diagnoses table
    """
    pending = get_pending_studies(study_id=study_id, split=split)

    if limit:
        pending = pending[:limit]

    print(f"Studies pending diagnosis: {len(pending)}")
    if not pending:
        print("Nothing to diagnose.")
        return

    if not DEFAULT_WEIGHTS_PATH.exists():
        print(
            f"ERROR: No classifier weights found at {DEFAULT_WEIGHTS_PATH}.\n"
            f"Run train_classifier.py first, or if this is the very first run\n"
            f"you can bootstrap with random weights by running train_classifier.py\n"
            f"with whatever data is available."
        )
        return

    store = FAISSStore()
    model = load_model(DEFAULT_WEIGHTS_PATH)
    model.eval()

    success = 0
    failed  = 0

    for i, row in enumerate(pending):
        sid = row["study_id"]

        try:
            # 1. Reconstruct image embedding from FAISS
            image_embedding = store.get_vector(row["faiss_id"])

            # 2. Deserialise report embedding from PostgreSQL BYTEA
            raw = row["report_embedding"]
            report_embedding = np.frombuffer(bytes(raw), dtype=np.float32).copy()

            if report_embedding.shape[0] != 128:
                print(f"  SKIP {sid}: unexpected report embedding dim {report_embedding.shape}")
                failed += 1
                continue

            # 3. Run classifier inference — returns {condition: probability}
            probabilities = predict(model, image_embedding, report_embedding)

            # 4. Persist to diagnoses table
            insert_diagnosis(sid, probabilities)

            success += 1

        except Exception as e:
            print(f"  ERROR on study {sid}: {e}")
            failed += 1
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(pending):
            print(
                f"  [{i+1}/{len(pending)}]  "
                f"success: {success}  failed: {failed}"
            )

    print(f"\nDone.")
    print(f"  Diagnosed : {success}")
    print(f"  Failed    : {failed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 5 — Diagnosis via MultimodalClassifier (CheXNet)"
    )
    parser.add_argument(
        "--study-id", type=str, default=None,
        help="Diagnose a single study by ID (for debugging)"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None,
        help="Restrict to one data split"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of studies processed this run"
    )
    args = parser.parse_args()

    run(study_id=args.study_id, split=args.split, limit=args.limit)