import subprocess
import gzip
import csv
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Get my secret stuff
load_dotenv()
PHYSIONET_USERNAME = os.getenv("PHYSIONET_USERNAME")
PHYSIONET_PASSWORD = os.getenv("PHYSIONET_PASSWORD")

# Config
RAW_FILES_ROOT = Path("data/raw/mimic-cxr/2.1.0/files")

# The official CheXpert labels live in the MIMIC-CXR-JPG dataset, who knew lol
CHEXPERT_URL  = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
CHEXPERT_PATH = Path("data/raw/mimic-cxr-2.0.0-chexpert.csv.gz")

# Official 14 CheXpert condition column names (as they appear in the CSV)
CONDITIONS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

WGET_AUTH = [f"--user={PHYSIONET_USERNAME}", f"--password={PHYSIONET_PASSWORD}"]


def download_chexpert_labels():
    if CHEXPERT_PATH.exists():
        print(f"CheXpert labels already present at {CHEXPERT_PATH}, skipping download.")
        return

    print("Downloading official CheXpert labels...")
    CHEXPERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "wget", "-N", "-c", "-q", "--show-progress",
        *WGET_AUTH,
        "-O", str(CHEXPERT_PATH),
        CHEXPERT_URL,
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


def find_studies(raw_files_root: Path) -> dict[str, dict]:
    studies = defaultdict(lambda: {"dcm_files": [], "subject_id": None, "study_id": None})

    for dcm_path in raw_files_root.rglob("*.dcm"):
        # Structure: files/p10/p10000032/s50414267/image.dcm
        parts     = dcm_path.parts
        p_folder  = parts[-3]
        s_folder  = parts[-2]
        study_key = f"{p_folder}/{s_folder}"

        studies[study_key]["dcm_files"].append(dcm_path)
        studies[study_key]["subject_id"] = p_folder[1:]
        studies[study_key]["study_id"]   = s_folder[1:]

    return studies


def print_label_distribution(rows: list[dict]):
    total = len(rows)
    print(f"\nLabel distribution ({total:,} images):")
    print(f"  {'Condition':<30} {'Positive':>8} {'Uncertain':>10} {'Negative':>9} {'Pos %':>7}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*9} {'-'*7}")
    for cond in CONDITIONS:
        positive  = sum(1 for r in rows if r[cond] == "1.0")
        uncertain = sum(1 for r in rows if r[cond] == "-1.0")
        negative  = sum(1 for r in rows if r[cond] == "0.0")
        pct       = 100 * positive / total if total else 0
        print(f"  {cond:<30} {positive:>8,} {uncertain:>10,} {negative:>9,} {pct:>6.1f}%")


def main():
    download_chexpert_labels()
    chexpert = load_chexpert_labels()

    print(f"\nScanning {RAW_FILES_ROOT} for DICOM files...")
    studies = find_studies(RAW_FILES_ROOT)
    print(f"Found {len(studies)} studies across "
          f"{sum(len(s['dcm_files']) for s in studies.values()):,} images.")

    no_label = [k for k, v in studies.items() if v["study_id"] not in chexpert]
    if no_label:
        print(f"\nWARNING: {len(no_label)} studies have no CheXpert label and will be skipped.")

    valid_studies = {k: v for k, v in studies.items() if v["study_id"] in chexpert}
    print(f"{len(valid_studies):,} studies have official labels.")

    rows = [
        {"study_id": study["study_id"], **chexpert[study["study_id"]]}
        for study in valid_studies.values()
        for _ in study["dcm_files"]
    ]

    print_label_distribution(rows)


if __name__ == "__main__":
    main()