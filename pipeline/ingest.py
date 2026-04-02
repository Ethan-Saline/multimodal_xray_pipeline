import csv
import argparse
from pathlib import Path

from storage.db import (
    start_pipeline_run, finish_pipeline_run,
    upsert_patient, upsert_study, insert_labels,
    get_cursor,
)
from storage.faiss_store import FAISSStore
from storage.cache import ping as redis_ping

# Manifest lives in data/raw — no processed folder
MANIFEST_PATH = Path("data/raw/manifest.csv")
REPORT_ROOT   = Path("data/raw/mimic-cxr/2.1.0/files")

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


def load_report_txt(subject_id: str, study_id: str) -> str | None:
    """Load the raw .txt radiology report for a study from data/raw."""
    prefix   = f"p{subject_id[:2]}"
    txt_path = REPORT_ROOT / prefix / f"p{subject_id}" / f"s{study_id}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def get_existing_study_ids() -> set[str]:
    """Return study_ids already in PostgreSQL — skip on re-run."""
    with get_cursor() as cur:
        cur.execute("SELECT study_id FROM studies")
        return {row["study_id"] for row in cur.fetchall()}


def run(manifest_path: Path):
    print("Checking services...")
    if redis_ping():
        print("  Redis ✓")
    else:
        print("  WARNING: Redis not reachable — cache unavailable this run")

    store = FAISSStore()
    print(f"  FAISS index ready (current size: {store.size})")

    run_id = start_pipeline_run()
    print(f"\nPipeline run id: {run_id}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Manifest: {len(rows)} rows")

    # Group by study — manifest has one row per DICOM
    studies_seen: dict[str, dict] = {}
    for row in rows:
        sid = row["study_id"]
        if sid not in studies_seen:
            studies_seen[sid] = row
    print(f"Unique studies in manifest: {len(studies_seen)}")

    # Skip studies already in PostgreSQL
    existing    = get_existing_study_ids()
    new_studies = {sid: row for sid, row in studies_seen.items() if sid not in existing}
    print(f"Already in PostgreSQL : {len(existing)}")
    print(f"New studies to ingest : {len(new_studies)}")

    if not new_studies:
        print("Nothing new to ingest.")
        finish_pipeline_run(run_id, studies_in=0, studies_out=0)
        return

    loaded = 0
    for study_id, row in new_studies.items():
        subject_id = row["subject_id"]
        split      = row["split"]

        upsert_patient(subject_id)
        upsert_study(
            study_id   = study_id,
            subject_id = subject_id,
            split      = split,
            run_id     = run_id,
            report_txt = load_report_txt(subject_id, study_id),
        )
        insert_labels(study_id, {c: row.get(c, "") for c in CONDITIONS})

        loaded += 1
        if loaded % 500 == 0:
            print(f"  {loaded}/{len(new_studies)} studies loaded...")

    finish_pipeline_run(run_id, studies_in=len(new_studies), studies_out=loaded)
    print(f"\nDone. {loaded} studies added to PostgreSQL.")
    print(f"FAISS index size: {store.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    run(args.manifest)
