import numpy as np
from storage.cache import cache_get, cache_set
from storage.db import get_cursor
from storage.faiss_store import FAISSStore

DEFAULT_K = 5
CACHE_TTL = 60 * 60 * 24   # 1 day

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


def _get_study_data(study_ids: list[str]) -> dict[str, dict]:
    """
    Fetch report text, mapped labels, and diagnosis probabilities
    for a list of study_ids in one round-trip.
    """
    if not study_ids:
        return {}

    with get_cursor() as cur:
        # Report text from studies table
        cur.execute(
            "SELECT study_id, report_txt FROM studies WHERE study_id = ANY(%s)",
            (study_ids,)
        )
        reports = {row["study_id"]: row["report_txt"] for row in cur.fetchall()}

        # Mapped labels
        cur.execute(
            """SELECT study_id, condition, mapped_value
                FROM label_mappings WHERE study_id = ANY(%s)""",
            (study_ids,)
        )
        labels: dict[str, dict] = {sid: {} for sid in study_ids}
        for row in cur.fetchall():
            labels[row["study_id"]][row["condition"]] = row["mapped_value"]

        # Diagnosis probabilities (may not exist yet — returns empty dict if missing)
        cur.execute(
            """SELECT study_id, condition, probability
                FROM diagnoses WHERE study_id = ANY(%s)""",
            (study_ids,)
        )
        diagnoses: dict[str, dict] = {sid: {} for sid in study_ids}
        for row in cur.fetchall():
            diagnoses[row["study_id"]][row["condition"]] = row["probability"]

    result = {}
    for sid in study_ids:
        result[sid] = {
            "study_id":   sid,
            "report_txt": reports.get(sid),
            "labels":     labels.get(sid, {}),
            "diagnoses":  diagnoses.get(sid, {}),
        }
    return result


def _faiss_id_to_study_id(faiss_ids: list[int]) -> dict[int, str]:
    """Bulk reverse lookup: faiss_id → study_id."""
    if not faiss_ids:
        return {}
    with get_cursor() as cur:
        cur.execute(
            "SELECT faiss_id, study_id FROM embeddings WHERE faiss_id = ANY(%s)",
            (faiss_ids,)
        )
        return {row["faiss_id"]: row["study_id"] for row in cur.fetchall()}


def retrieve(
    embedding: np.ndarray,
    k: int = DEFAULT_K,
    exclude_study_id: str = None,
) -> list[dict]:
    """
    Find the K most similar cases to the given embedding.

    Parameters
    ----------
    embedding        : 128-dim float32 numpy array (BioViL output)
    k                : number of similar cases to return
    exclude_study_id : exclude the query study itself from results

    Returns
    -------
    List of dicts sorted by similarity (closest first):
    [
        {
            "study_id"  : "50414267",
            "faiss_id"  : 3,
            "distance"  : 0.021,
            "report_txt": "Findings: ...",
            "labels"    : {"Pneumonia": 1, "Effusion": 0, ...},
            "diagnoses" : {"Pneumonia": 0.83, "Effusion": 0.11, ...},
        },
        ...
    ]
    """
    cache_key = f"retrieval:{hash(embedding.tobytes())}:k{k}"
    cached    = cache_get(cache_key)
    if cached is not None:
        return cached

    store    = FAISSStore()
    search_k = k + 1 if exclude_study_id else k
    raw      = store.search(embedding, k=search_k)

    if not raw:
        return []

    faiss_ids = [r["faiss_id"] for r in raw]
    id_map    = _faiss_id_to_study_id(faiss_ids)

    kept = []
    for r in raw:
        sid = id_map.get(r["faiss_id"])
        if sid is None:
            continue
        if exclude_study_id and sid == exclude_study_id:
            continue
        kept.append({"study_id": sid, "faiss_id": r["faiss_id"], "distance": r["distance"]})
        if len(kept) == k:
            break

    study_ids  = [r["study_id"] for r in kept]
    study_data = _get_study_data(study_ids)

    results = []
    for r in kept:
        data = study_data.get(r["study_id"], {})
        results.append({
            "study_id":   r["study_id"],
            "faiss_id":   r["faiss_id"],
            "distance":   r["distance"],
            "report_txt": data.get("report_txt"),
            "labels":     data.get("labels", {}),
            "diagnoses":  data.get("diagnoses", {}),
        })

    cache_set(cache_key, results, ttl=CACHE_TTL)
    return results


# ------------------------------------------------------------------ #
# Smoke test — python -m pipeline.retrieve                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Running retriever smoke test...\n")

    with get_cursor() as cur:
        cur.execute("SELECT study_id, dicom_uid, faiss_id FROM embeddings LIMIT 1")
        row = cur.fetchone()

    if row is None:
        print("No embeddings found — run encode.py first.")
        exit(1)

    study_id = row["study_id"]
    faiss_id = row["faiss_id"]
    print(f"Query study: {study_id}  faiss_id: {faiss_id}")

    # Reconstruct embedding from FAISS
    store     = FAISSStore()
    embedding = store.get_vector(faiss_id)
    print(f"  Embedding shape: {embedding.shape} ✓")

    results = retrieve(embedding, k=5, exclude_study_id=study_id)
    print(f"\nTop {len(results)} similar cases:\n")

    for i, r in enumerate(results, 1):
        positive = [c for c, v in r["labels"].items() if v == 1]
        print(f"  {i}. {r['study_id']}  distance={r['distance']:.4f}")
        print(f"     Labels: {', '.join(positive) if positive else 'none'}")
        if r["diagnoses"]:
            top_diag = sorted(r["diagnoses"].items(), key=lambda x: -x[1])[:3]
            print(f"     Diagnoses: {', '.join(f'{c}={p:.2f}' for c, p in top_diag)}")
        if r["report_txt"]:
            print(f"     Report: {r['report_txt'][:100].replace(chr(10), ' ')}...")
        print()

    print("Retriever smoke test passed.")
