import argparse
import warnings

import numpy as np
import torch
from health_multimodal.text.utils import get_biovil_t_bert

from storage.cache import cache_embedding, get_cached_embedding
from storage.db import get_cursor, insert_report
from storage.faiss_store import FAISSStore
from pipeline.retrieve import retrieve

warnings.filterwarnings("ignore", category=FutureWarning)

REPORT_CACHE_TTL = 60 * 60 * 24 * 7   # 7 days
DEFAULT_K        = 5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_biovil_t():
    """Load BioViL-T text model. Returns (tokenizer, model) tuple."""
    print("Loading BioViL-T model...")
    tokenizer, model = get_biovil_t_bert()
    model.eval()
    print("BioViL-T ready.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Core report generation
# ---------------------------------------------------------------------------

def generate_report(
    text_model,
    image_embedding: np.ndarray,
    retrieved_cases: list[dict],
) -> dict:
    """
    Synthesize a report from retrieved cases and encode it into a
    128-dim projected text embedding using BioViL-T.

    The projected CLS embedding lives in the same space as the BioViL
    image embedding, keeping the 256-dim concatenation
    (128 image + 128 report) clean for the downstream classifier.

    Parameters
    ----------
    text_model      : (tokenizer, model) tuple from load_biovil_t()
    image_embedding : 128-dim float32 BioViL image embedding
    retrieved_cases : output of retrieve.retrieve()

    Returns
    -------
    {
        "findings"        : str,
        "impression"      : str,
        "report_embedding": np.ndarray (128-dim float32),
        "full_report"     : str,
    }
    """
    tokenizer, model = text_model

    # Synthesize report text from retrieved cases
    positive_conditions = set()
    report_snippets     = []
    for case in retrieved_cases:
        for c, v in case.get("labels", {}).items():
            if v == 1:
                positive_conditions.add(c)
        if case.get("report_txt"):
            report_snippets.append(case["report_txt"][:200].replace("\n", " ").strip())

    conditions_str = ", ".join(sorted(positive_conditions)) if positive_conditions else "No Finding"
    findings    = f"Findings consistent with retrieved similar cases. Conditions noted: {conditions_str}."
    impression  = report_snippets[0] if report_snippets else "No acute cardiopulmonary process."
    full_report = f"FINDINGS: {findings}\nIMPRESSION: {impression}"

    # Encode report text into 128-dim projected embedding.
    # output_cls_projected_embedding=True activates the projection head
    # that maps 768-dim BERT CLS → 128-dim shared image/text space.
    with torch.no_grad():
        inputs = tokenizer(
            full_report,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs          = model(**inputs, output_cls_projected_embedding=True)
        report_embedding = outputs.cls_projected_embedding          # 128-dim
        report_embedding = torch.nn.functional.normalize(report_embedding, dim=-1)
        report_embedding = report_embedding.squeeze(0).cpu().numpy().astype(np.float32)

    return {
        "findings":         findings,
        "impression":       impression,
        "report_embedding": report_embedding,
        "full_report":      full_report,
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_studies_pending_report() -> list[dict]:
    """
    Return studies that have an image embedding but no generated report yet.
    Joins embeddings → studies, filters out studies already in reports table.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT e.study_id, e.faiss_id
            FROM   embeddings e
            LEFT   JOIN reports r ON r.study_id = e.study_id
            WHERE  r.study_id IS NULL
            ORDER  BY e.study_id
            """
        )
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(study_id: str = None, limit: int = None, split: str = None):
    """
    For each study that has an image embedding but no report:
      1. Load image embedding from FAISS
      2. Retrieve top-5 similar historical cases
      3. Synthesize findings + impression from retrieved context
      4. Encode the report into a 128-dim projected text embedding
      5. Persist report text and embedding to PostgreSQL
      6. Cache report embedding in Redis via pickle (not JSON)
    """
    pending = get_studies_pending_report()

    if study_id:
        pending = [p for p in pending if p["study_id"] == study_id]
    if split:
        with get_cursor() as cur:
            cur.execute("SELECT study_id FROM studies WHERE split = %s", (split,))
            split_ids = {row["study_id"] for row in cur.fetchall()}
        pending = [p for p in pending if p["study_id"] in split_ids]
    if limit:
        pending = pending[:limit]

    print(f"Studies pending report generation: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    store      = FAISSStore()
    text_model = load_biovil_t()

    success = 0
    failed  = 0
    skipped = 0

    for i, row in enumerate(pending):
        sid      = row["study_id"]
        faiss_id = row["faiss_id"]

        # Cache key uses "report:" prefix; stored via pickle to handle numpy arrays
        cache_key = f"report:{sid}"
        if get_cached_embedding(cache_key) is not None:
            skipped += 1
            continue

        try:
            # 1. Reconstruct image embedding from FAISS
            image_embedding = store.get_vector(faiss_id)

            # 2. Retrieve top-K similar historical cases
            retrieved = retrieve(
                embedding        = image_embedding,
                k                = DEFAULT_K,
                exclude_study_id = sid,
            )

            # 3 & 4. Synthesize report text + encode to 128-dim embedding
            result = generate_report(text_model, image_embedding, retrieved)

            # 5. Persist to PostgreSQL
            insert_report(
                study_id         = sid,
                findings         = result["findings"],
                impression       = result["impression"],
                full_report      = result["full_report"],
                report_embedding = result["report_embedding"],
            )

            # 6. Cache report embedding using pickle (numpy-safe)
            cache_embedding(cache_key, result["report_embedding"], ttl=REPORT_CACHE_TTL)

            success += 1

        except Exception as e:
            print(f"  ERROR on study {sid}: {e}")
            failed += 1
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            print(
                f"  [{i+1}/{len(pending)}]  "
                f"success: {success}  failed: {failed}  skipped(cached): {skipped}"
            )

    print(f"\nDone.")
    print(f"  Generated : {success}")
    print(f"  Failed    : {failed}")
    print(f"  Skipped   : {skipped}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6 — Report generation via BioViL-T")
    parser.add_argument("--study-id", type=str,  default=None,
                        help="Process a single study by ID (for debugging)")
    parser.add_argument("--limit",    type=int,  default=None,
                        help="Cap the number of studies processed this run")
    parser.add_argument("--split",    choices=["train", "val", "test"], default=None,
                        help="Restrict to a specific data split")
    args = parser.parse_args()

    run(study_id=args.study_id, limit=args.limit, split=args.split)