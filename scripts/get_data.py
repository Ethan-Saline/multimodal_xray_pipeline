import subprocess
import os
import gzip
import csv
import time
from pathlib import Path
from dotenv import load_dotenv

# Get secret stuff from .env
load_dotenv()
PHYSIONET_USERNAME = os.getenv("PHYSIONET_USERNAME")
PHYSIONET_PASSWORD = os.getenv("PHYSIONET_PASSWORD")

# Config
BASE_URL   = "https://physionet.org/files/mimic-cxr/2.1.0/"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FILES_ROOT = OUTPUT_DIR / "mimic-cxr" / "2.1.0" / "files"

# How many NEW images to grab this run
NUM_NEW_IMAGES = 5
MAX_RETRIES    = 5
RETRY_WAIT_SEC = 30

WGET_AUTH = [f"--user={PHYSIONET_USERNAME}", f"--password={PHYSIONET_PASSWORD}"]


# ------------------------------------------------------------------ #
# wget helpers                                                         #
# ------------------------------------------------------------------ #

def wget_single(url: str, dest: Path):
    subprocess.run([
        "wget", "-N", "-c", "-q", "--show-progress",
        *WGET_AUTH, "-P", str(dest), url,
    ], check=True)


def wget_batch(subset_path: Path, output_dir: Path):
    for attempt in range(1, MAX_RETRIES + 1):
        result = subprocess.run([
            "wget", "-r", "-N", "-c", "-np", "-nH",
            "--cut-dirs=1",
            "--tries=3",
            "--waitretry=10",
            "--timeout=60",
            *WGET_AUTH,
            "-i", str(subset_path),
            f"--base={BASE_URL}",
            "-P", str(output_dir),
        ])

        if result.returncode == 0:
            print("wget completed successfully.")
            return

        if result.returncode in (4, 8):
            if attempt < MAX_RETRIES:
                print(f"\nNetwork error (exit {result.returncode}). "
                    f"Waiting {RETRY_WAIT_SEC}s before retry {attempt}/{MAX_RETRIES - 1}...")
                time.sleep(RETRY_WAIT_SEC)
                print("Resuming download...")
            else:
                print(f"\nFailed after {MAX_RETRIES} attempts.")
                raise SystemExit(1)
        else:
            raise subprocess.CalledProcessError(result.returncode, result.args)


# ------------------------------------------------------------------ #
# Find already-downloaded studies                                      #
# ------------------------------------------------------------------ #

def get_existing_study_ids() -> set[str]:
    """
    Scan FILES_ROOT for study folders that already exist locally.
    Returns a set of study_id strings (without the 's' prefix).
    """
    existing = set()
    if not FILES_ROOT.exists():
        return existing
    for study_dir in FILES_ROOT.rglob("s*"):
        if study_dir.is_dir():
            existing.add(study_dir.name[1:])   # strip leading 's'
    return existing


# ------------------------------------------------------------------ #
# Download record list                                                 #
# ------------------------------------------------------------------ #

record_list_gz = OUTPUT_DIR / "cxr-record-list.csv.gz"

if not record_list_gz.exists():
    print("Downloading cxr-record-list.csv.gz ...")
    wget_single(BASE_URL + "cxr-record-list.csv.gz", OUTPUT_DIR)
else:
    print("cxr-record-list.csv.gz already present, skipping.")


# ------------------------------------------------------------------ #
# Find existing studies and pick up where we left off                  #
# ------------------------------------------------------------------ #

print("Scanning for already-downloaded studies...")
existing_study_ids = get_existing_study_ids()
print(f"  Found {len(existing_study_ids)} studies already on disk.")

print(f"Parsing record list, looking for {NUM_NEW_IMAGES} new studies...")

image_urls = []
skipped    = 0

with gzip.open(record_list_gz, "rt") as f:
    reader = csv.DictReader(f)
    for row in reader:
        study_id   = row["study_id"]
        subject_id = row["subject_id"]
        dicom_id   = row["dicom_id"]

        # Skip studies we already have
        if study_id in existing_study_ids:
            skipped += 1
            continue

        p_group   = "p" + subject_id[:2]
        p_folder  = "p" + subject_id
        s_folder  = "s" + study_id
        image_url = f"files/{p_group}/{p_folder}/{s_folder}/{dicom_id}.dcm"

        image_urls.append((image_url, subject_id, study_id))

        if len(image_urls) >= NUM_NEW_IMAGES:
            break

print(f"  Skipped {skipped} already-downloaded entries.")
print(f"  New images to download: {len(image_urls)}")

if not image_urls:
    print("Nothing new to download.")
    raise SystemExit(0)


# ------------------------------------------------------------------ #
# Build download list (images + reports)                               #
# ------------------------------------------------------------------ #

download_urls = []
seen_reports  = set()

for image_url, subject_id, study_id in image_urls:
    download_urls.append(image_url)

    p_group    = "p" + subject_id[:2]
    p_folder   = "p" + subject_id
    report_url = f"files/{p_group}/{p_folder}/s{study_id}.txt"

    if report_url not in seen_reports:
        download_urls.append(report_url)
        seen_reports.add(report_url)

print(f"Total files to download: {len(download_urls)} "
    f"({len(image_urls)} images + {len(seen_reports)} reports)")


# ------------------------------------------------------------------ #
# Download                                                             #
# ------------------------------------------------------------------ #

subset_path = OUTPUT_DIR / "download_subset.txt"
subset_path.write_text("\n".join(download_urls) + "\n")

print(f"\nDownloading to {OUTPUT_DIR} ...")
wget_batch(subset_path, OUTPUT_DIR)
subset_path.unlink()
print("Download complete!")


# ------------------------------------------------------------------ #
# Verify                                                               #
# ------------------------------------------------------------------ #

print("\nVerifying image-report pairing...")
missing = []
for image_url, subject_id, study_id in image_urls:
    p_group      = "p" + subject_id[:2]
    p_folder     = "p" + subject_id
    report_local = FILES_ROOT / p_group / p_folder / f"s{study_id}.txt"
    if not report_local.exists():
        missing.append(report_local)

unique_missing = sorted(set(str(p) for p in missing))
if unique_missing:
    print(f"WARNING: {len(unique_missing)} studies missing their report:")
    for path in unique_missing:
        print(f"  {path}")
else:
    print(f"All {len(image_urls)} new images have a paired report.")
    print(f"Total studies on disk: {len(existing_study_ids) + len(image_urls)}")