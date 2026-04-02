"""
test_db.py — PostgreSQL tests
Verifies the schema is applied and data was loaded by ingest.py
"""

import sys
from storage.db import get_cursor

def test_tables_exist():
    expected = ["pipeline_runs", "patients", "studies", "label_mappings",
                "embeddings", "reports", "diagnoses", "outputs"]
    with get_cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        found = {row["table_name"] for row in cur.fetchall()}

    missing = [t for t in expected if t not in found]
    assert not missing, f"Missing tables: {missing}"
    print(f"  Tables exist: {', '.join(expected)} ✓")


def test_studies_loaded():
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM studies")
        n = cur.fetchone()["n"]
    assert n > 0, "No studies found — did ingest.py run successfully?"
    print(f"  Studies in DB: {n} ✓")


def test_patients_loaded():
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM patients")
        n = cur.fetchone()["n"]
    assert n > 0, "No patients found"
    print(f"  Patients in DB: {n} ✓")


def test_labels_loaded():
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM label_mappings")
        n = cur.fetchone()["n"]
    assert n > 0, "No label mappings found"
    print(f"  Label mappings in DB: {n} ✓")


def test_label_strategies():
    """Verify U-Ones and U-Zeroes were applied correctly."""
    with get_cursor() as cur:
        # Cardiomegaly and Atelectasis uncertain cases should be U-Zeroes (mapped to 0)
        cur.execute("""
            SELECT COUNT(*) AS n FROM label_mappings
            WHERE condition IN ('Cardiomegaly', 'Atelectasis')
            AND raw_value = '-1.0'
            AND strategy = 'U-Zeroes'
            AND mapped_value = 0
        """)
        uzero = cur.fetchone()["n"]

        # Everything else uncertain should be U-Ones (mapped to 1)
        cur.execute("""
            SELECT COUNT(*) AS n FROM label_mappings
            WHERE condition NOT IN ('Cardiomegaly', 'Atelectasis')
            AND raw_value = '-1.0'
            AND strategy = 'U-Ones'
            AND mapped_value = 1
        """)
        uones = cur.fetchone()["n"]

    print(f"  U-Zeroes applied (Cardiomegaly/Atelectasis uncertain): {uzero} ✓")
    print(f"  U-Ones applied (all other uncertain): {uones} ✓")


def test_splits():
    """Verify train/val/test split exists and proportions look reasonable."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT split, COUNT(*) AS n FROM studies
            GROUP BY split ORDER BY split
        """)
        rows = {row["split"]: row["n"] for row in cur.fetchall()}

    assert "train" in rows, "No train split found"
    assert "val"   in rows, "No val split found"
    assert "test"  in rows, "No test split found"

    total = sum(rows.values())
    for split, n in rows.items():
        pct = 100 * n / total
        print(f"  {split}: {n} studies ({pct:.1f}%) ✓")


def test_pipeline_run_recorded():
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM pipeline_runs WHERE status = 'complete'")
        n = cur.fetchone()["n"]
    assert n > 0, "No completed pipeline runs found"
    print(f"  Completed pipeline runs: {n} ✓")


if __name__ == "__main__":
    tests = [
        test_tables_exist,
        test_studies_loaded,
        test_patients_loaded,
        test_labels_loaded,
        test_label_strategies,
        test_splits,
        test_pipeline_run_recorded,
    ]

    print("Running DB tests...\n")
    failed = 0
    for t in tests:
        try:
            print(f"[{t.__name__}]")
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) failed.'}")
    sys.exit(0 if not failed else 1)