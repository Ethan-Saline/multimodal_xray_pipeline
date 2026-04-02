import sys
import tempfile
import numpy as np
from pathlib import Path
from storage.faiss_store import FAISSStore


def test_index_initialises():
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSStore(index_path=Path(tmp) / "test.bin", dim=16)
        assert store.size == 0
        print("  Empty index initialises ✓")


def test_add_single():
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSStore(index_path=Path(tmp) / "test.bin", dim=16)
        vec = np.random.rand(16).astype(np.float32)
        fid = store.add(vec)
        assert fid == 0
        assert store.size == 1
        print(f"  Single add → faiss_id={fid} ✓")


def test_add_batch():
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSStore(index_path=Path(tmp) / "test.bin", dim=16)
        vecs = np.random.rand(10, 16).astype(np.float32)
        ids  = store.add_batch(vecs)
        assert ids == list(range(10))
        assert store.size == 10
        print(f"  Batch add → faiss_ids 0-9 ✓")


def test_search_returns_self():
    """A vector's nearest neighbour should be itself."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSStore(index_path=Path(tmp) / "test.bin", dim=16)
        vecs  = np.random.rand(5, 16).astype(np.float32)
        store.add_batch(vecs)

        results = store.search(vecs[0], k=3)
        assert results[0]["faiss_id"] == 0
        assert results[0]["distance"] < 1e-5
        print(f"  Search nearest neighbour is self ✓")


def test_search_ordering():
    """Results should be sorted closest first."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSStore(index_path=Path(tmp) / "test.bin", dim=16)
        vecs  = np.random.rand(20, 16).astype(np.float32)
        store.add_batch(vecs)

        results = store.search(vecs[0], k=5)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances), "Results not sorted by distance"
        print(f"  Search results sorted by distance ✓")


def test_persist_and_reload():
    """Index should survive a reload from disk."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.bin"
        store = FAISSStore(index_path=path, dim=16)
        vecs  = np.random.rand(7, 16).astype(np.float32)
        store.add_batch(vecs)

        store2 = FAISSStore(index_path=path, dim=16)
        assert store2.size == 7
        print(f"  Persisted and reloaded, size={store2.size} ✓")


def test_production_index_exists():
    """Check the actual index file created by ingest.py exists."""
    index_path = Path("data/faiss/index.bin")
    assert index_path.exists(), f"Production index not found at {index_path}"
    store = FAISSStore()
    print(f"  Production index exists, size={store.size} ✓")


if __name__ == "__main__":
    tests = [
        test_index_initialises,
        test_add_single,
        test_add_batch,
        test_search_returns_self,
        test_search_ordering,
        test_persist_and_reload,
        test_production_index_exists,
    ]

    print("Running FAISS tests...\n")
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
    