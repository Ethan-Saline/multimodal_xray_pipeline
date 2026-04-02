import subprocess
import gzip
import csv
import random
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
PHYSIONET_USERNAME = os.getenv("PHYSIONET_USERNAME")
PHYSIONET_PASSWORD = os.getenv("PHYSIONET_PASSWORD")

RAW_FILES_ROOT = Path("data/raw/mimic-cxr/2.1.0/files")
MANIFEST_PATH  = Path("data/raw/manifest.csv")   # lives in data/raw, not processed

CHEXPERT_URL  = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
CHEXPERT_PATH = Path("data/raw/mimic-cxr-2.0.0-chexpert.csv.gz")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

WGET_AUTH = [f"--user={PHYSIONET_USERNAME}", f"--password={PHYSIONET_PASSWORD}"]


def download_chexpert_labels():
    if CHEXPERT_PATH.exists():
        print("CheXpert labels already present, skipping download.")
        return
    print("Downloading official CheXpert labels...")
    CHEXPERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "wget", "-N", "-c", "-q", "--show-progress",
        *WGET_AUTH, "-O", str(CHEXPERT_PATH), CHEXPERT_URL,
    ], check=True)
    print("Downloaded.")


def load_chexpert_labels() -> dict[str, dict]:
    labels = {}
    print("Loading CheXpert labels...")
    with gzip.open(CHEXPERT_PATH, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_id = row["study_id"]
            labels[study_id] = {cond: row.get(cond, "") for cond in CONDITIONS}
    print(f"Loaded labels for {len(labels)} studies.")
    return labels


def load_existing_manifest() -> tuple[dict[str, str], set[str]]:
    """
    Load existing manifest to preserve split assignments.
    Returns (existing_splits dict, existing_study_ids set).
    """
    existing_splits    = {}
    existing_study_ids = set()

    if not MANIFEST_PATH.exists():
        return existing_splits, existing_study_ids

    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["study_id"]
            existing_splits[sid]   = row["split"]
            existing_study_ids.add(sid)

    print(f"Existing manifest: {len(existing_study_ids)} studies already processed.")
    return existing_splits, existing_study_ids


def find_studies(raw_files_root: Path) -> dict[str, dict]:
    studies = defaultdict(lambda: {"dcm_files": [], "subject_id": None, "study_id": None})
    for dcm_path in raw_files_root.rglob("*.dcm"):
        parts      = dcm_path.parts
        p_folder   = parts[-3]
        s_folder   = parts[-2]
        study_key  = f"{p_folder}/{s_folder}"
        studies[study_key]["dcm_files"].append(dcm_path)
        studies[study_key]["subject_id"] = p_folder[1:]
        studies[study_key]["study_id"]   = s_folder[1:]
    return studies


def assign_splits_incremental(
    new_study_keys: list[str],
    existing_splits: dict[str, str],
    ratios: dict,
    seed: int,
) -> dict[str, str]:
    """Assign splits to new studies only. Existing splits are unchanged."""
    if not new_study_keys:
        return existing_splits.copy()

    keys = sorted(new_study_keys)
    rng  = random.Random(seed + len(existing_splits))
    rng.shuffle(keys)

    n       = len(keys)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])

    new_splits = {}
    for i, key in enumerate(keys):
        if i < n_train:
            new_splits[key] = "train"
        elif i < n_train + n_val:
            new_splits[key] = "val"
        else:
            new_splits[key] = "test"

    return {**new_splits, **existing_splits}


def main():
    download_chexpert_labels()
    chexpert = load_chexpert_labels()

    existing_splits, existing_study_ids = load_existing_manifest()

    print(f"Scanning {RAW_FILES_ROOT} for DICOM files...")
    studies = find_studies(RAW_FILES_ROOT)
    print(f"Found {len(studies)} studies across "
          f"{sum(len(s['dcm_files']) for s in studies.values())} images.")

    no_label = [k for k, v in studies.items() if v["study_id"] not in chexpert]
    if no_label:
        print(f"WARNING: {len(no_label)} studies have no CheXpert label and will be skipped.")

    valid_studies = {k: v for k, v in studies.items() if v["study_id"] in chexpert}

    new_study_keys = [
        k for k, v in valid_studies.items()
        if v["study_id"] not in existing_study_ids
    ]
    print(f"Already processed : {len(existing_study_ids)}")
    print(f"New studies       : {len(new_study_keys)}")

    if not new_study_keys:
        print("No new studies to process.")
        return

    all_splits = assign_splits_incremental(
        new_study_keys, existing_splits, SPLIT_RATIOS, RANDOM_SEED
    )

    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in all_splits.values():
        split_counts[s] += 1
    print(f"Split totals: train={split_counts['train']}  "
          f"val={split_counts['val']}  test={split_counts['test']}")

    # Build new manifest rows — dicom_path points directly at data/raw
    new_manifest_rows = []
    skipped = 0

    print("Building manifest rows...")
    for study_key in new_study_keys:
        study      = valid_studies[study_key]
        split      = all_splits[study_key]
        study_id   = study["study_id"]
        subject_id = study["subject_id"]
        labels     = chexpert[study_id]

        for dcm_path in study["dcm_files"]:
            new_manifest_rows.append({
                "split":      split,
                "subject_id": subject_id,
                "study_id":   study_id,
                "dicom_path": str(dcm_path),   # direct path in data/raw — no copy
                **labels,
            })

    # Append to manifest — never overwrite existing rows
    fieldnames   = ["split", "subject_id", "study_id", "dicom_path"] + CONDITIONS
    write_header = not MANIFEST_PATH.exists()

    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_manifest_rows)

    print(f"\nDone!")
    print(f"  New rows added : {len(new_manifest_rows)}")
    print(f"  Skipped        : {skipped}")
    print(f"  Manifest total : {len(existing_study_ids) + len(new_study_keys)} studies")

    print(f"\nLabel distribution for new studies:")
    for cond in CONDITIONS:
        positive  = sum(1 for r in new_manifest_rows if r[cond] == "1.0")
        uncertain = sum(1 for r in new_manifest_rows if r[cond] == "-1.0")
        pct = 100 * positive / len(new_manifest_rows) if new_manifest_rows else 0
        print(f"  {cond:<30} {positive:>5} positive  {uncertain:>5} uncertain  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
