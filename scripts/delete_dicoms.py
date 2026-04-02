"""
delete_dicoms.py — Safe DICOM deletion after pipeline is confirmed working
scripts/ | CheXpert Pipeline

Run this ONLY after:
  1. encode.py has completed for all studies
  2. feature_map column is populated in the embeddings table
  3. diagnose.py and validate.py have been verified end-to-end

Deletes DICOMs from data/raw and marks them deleted in PostgreSQL.
Dry-run mode is on by default — pass --confirm to actually delete.

Usage:
    python -m scripts.delete_dicoms              # dry run, shows what would be deleted
    python -m scripts.delete_dicoms --confirm    # actually deletes
    python -m scripts.delete_dicoms --split val --confirm
"""

import argparse
import csv
from pathlib import Path

from storage.db import get_cursor, mark_dicom_deleted

MANIFEST_PATH = Path("data/raw/manifest.csv")


def get_encoded_study_ids() -> set[str]:
    """Return study_ids that have a feature_map stored (safe to delete)."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT DISTINCT study_id FROM embeddings WHERE feature_map IS NOT NULL"
        )
        return {row["study_id"] for row in cur.fetchall()}


def get_already_deleted_uids() -> set[str]:
    """Return dicom_uids already marked deleted in PostgreSQL."""
    with get_cursor() as cur:
        cur.execute("SELECT dicom_uid FROM embeddings WHERE dicom_deleted = TRUE")
        return {row["dicom_uid"] for row in cur.fetchall()}


def run(split: str = None, confirm: bool = False):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    # Build study → dicom_paths mapping
    studies: dict[str, dict] = {}
    for row in rows:
        sid = row["study_id"]
        if split and row["split"] != split:
            continue
        if sid not in studies:
            studies[sid] = {
                "study_id":    sid,
                "subject_id":  row["subject_id"],
                "split":       row["split"],
                "dicom_paths": [],
                "dicom_uid":   None,
            }
        p = Path(row["dicom_path"])
        studies[sid]["dicom_paths"].append(p)

    encoded_ids      = get_encoded_study_ids()
    already_deleted  = get_already_deleted_uids()

    # Only delete studies that are fully encoded (feature_map present)
    # and not already deleted
    candidates = [
        s for s in studies.values()
        if s["study_id"] in encoded_ids
        and s["study_id"] not in already_deleted
    ]

    not_encoded = [
        s["study_id"] for s in studies.values()
        if s["study_id"] not in encoded_ids
    ]

    print(f"Studies in manifest     : {len(studies)}")
    print(f"Fully encoded           : {len(encoded_ids)}")
    print(f"Already deleted         : {len(already_deleted)}")
    print(f"Candidates for deletion : {len(candidates)}")

    if not_encoded:
        print(f"\nWARNING: {len(not_encoded)} studies have no feature_map and will be SKIPPED:")
        for sid in not_encoded[:10]:
            print(f"  {sid}")
        if len(not_encoded) > 10:
            print(f"  ... and {len(not_encoded) - 10} more")

    if not candidates:
        print("\nNothing to delete.")
        return

    if not confirm:
        print(f"\nDRY RUN — would delete DICOMs for {len(candidates)} studies.")
        total_files = sum(
            len([p for p in s["dicom_paths"] if p.exists()])
            for s in candidates
        )
        total_size = sum(
            p.stat().st_size
            for s in candidates
            for p in s["dicom_paths"]
            if p.exists()
        )
        print(f"  Files on disk : {total_files}")
        print(f"  Disk reclaimed: {total_size / 1e9:.2f} GB")
        print(f"\nRe-run with --confirm to proceed.")
        return

    print(f"\nDeleting DICOMs for {len(candidates)} studies...")
    deleted_files = 0
    failed        = 0

    for study in candidates:
        study_id  = study["study_id"]
        paths     = study["dicom_paths"]

        # Determine dicom_uid the same way encode.py does
        existing_stems = [p.stem for p in paths if p.exists()]
        dicom_uid = study_id if len(paths) > 1 else (existing_stems[0] if existing_stems else study_id)

        try:
            for p in paths:
                if p.exists():
                    p.unlink()
                    deleted_files += 1

            mark_dicom_deleted(dicom_uid)

        except Exception as e:
            print(f"  ERROR on study {study_id}: {e}")
            failed += 1

    print(f"\nDone.")
    print(f"  DICOM files deleted : {deleted_files}")
    print(f"  Studies failed      : {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete DICOMs after pipeline is verified end-to-end."
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete files. Without this flag, runs as a dry run."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None,
        help="Restrict deletion to one split."
    )
    args = parser.parse_args()
    run(split=args.split, confirm=args.confirm)