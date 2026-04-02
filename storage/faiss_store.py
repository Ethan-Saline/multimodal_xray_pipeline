import os
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss/index.bin"))
DEFAULT_DIM        = int(os.getenv("FAISS_DIM", "1024"))


class FAISSStore:
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH, dim: int = DEFAULT_DIM):
        self.index_path = Path(index_path)
        self.dim        = dim
        self.index      = self._load_or_create()

    def _load_or_create(self) -> faiss.Index:
        if self.index_path.exists():
            print(f"[FAISS] Loading index from {self.index_path}")
            index = faiss.read_index(str(self.index_path))
            print(f"[FAISS] Loaded. Vectors in index: {index.ntotal}")
            return index
        print(f"[FAISS] No index found — creating new FlatL2 (dim={self.dim})")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        return faiss.IndexFlatL2(self.dim)

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embedding: np.ndarray) -> int:
        """
        Add one embedding. Returns the faiss_id assigned to it.
        Store this id in the PostgreSQL embeddings table.
        """
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss_id = self.index.ntotal
        self.index.add(vec)
        self._save()
        return faiss_id

    def add_batch(self, embeddings: np.ndarray) -> list[int]:
        """
        Add a batch of embeddings. Returns list of faiss_ids.
        embeddings shape: (N, dim) float32
        """
        vecs     = embeddings.astype(np.float32)
        start_id = self.index.ntotal
        self.index.add(vecs)
        self._save()
        return list(range(start_id, self.index.ntotal))

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        """
        Find k nearest neighbours to query vector.
        Returns [{"faiss_id": int, "distance": float}, ...]
        sorted closest first.

        After getting faiss_ids, use db.get_study_id_by_faiss()
        to resolve them back to study data.
        """
        vec = query.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:   # fewer than k vectors exist
                continue
            results.append({"faiss_id": int(idx), "distance": float(dist)})
        return results

    def get_vector(self, faiss_id: int) -> np.ndarray:
        """Reconstruct a stored vector by faiss_id. Only works with FlatL2."""
        vec = np.zeros(self.dim, dtype=np.float32)
        self.index.reconstruct(faiss_id, vec)
        return vec


# ------------------------------------------------------------------ #
# Smoke test — python faiss_store.py                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import tempfile

    print("Running FAISS smoke test...")
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSStore(index_path=Path(tmp) / "test.bin", dim=8)

        vecs = np.random.rand(5, 8).astype(np.float32)
        ids  = store.add_batch(vecs)
        print(f"  Added {len(ids)} vectors → faiss_ids: {ids}")
        assert store.size == 5

        results = store.search(vecs[0], k=3)
        print(f"  Search results (k=3): {results}")
        assert results[0]["faiss_id"] == 0, "Closest should be itself"

        # Reload from disk
        store2 = FAISSStore(index_path=Path(tmp) / "test.bin", dim=8)
        assert store2.size == 5
        print(f"  Reloaded from disk, size={store2.size} ✓")

    print("FAISS smoke test passed.")
