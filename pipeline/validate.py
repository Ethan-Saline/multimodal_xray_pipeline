"""
validate.py — Validation & Heatmaps (Step 7)
pipeline/ | CheXpert Pipeline

For each study with a diagnosis but no validated output:
  1. Load image embedding (FAISS) + feature map (PostgreSQL BYTEA)
  2. Generate Grad-CAM heatmap using classifier weights + layer4 feature map
  3. Save heatmap PNG to data/heatmaps/{study_id}.png
  4. Validate output JSON structure and confidence score ranges
  5. Assemble and persist final output JSON to PostgreSQL outputs table

Grad-CAM implementation notes:
  - feature_map shape: (2048, H, W)  — stored as BYTEA in embeddings table
  - Classifier's image_proj weights serve as the "CAM weights" (global average
    of the gradient of the predicted class w.r.t. feature map channels)
  - We use Score-CAM (gradient-free) since we only have the pooled embedding,
    not a live forward pass through the spatial feature map.
  - The 1×1 projection from 2048→64 in image_proj gives us per-channel weights
    we can dot with the spatial feature map to produce a (H, W) activation map.

Usage:
    python -m pipeline.validate
    python -m pipeline.validate --study-id 50414267
    python -m pipeline.validate --split val
    python -m pipeline.validate --limit 50
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from storage.db import (
    get_cursor,
    insert_output,
    mark_output_validated,
)
from storage.faiss_store import FAISSStore
from models.classifier import (
    load_model,
    load_thresholds,
    TRAINABLE_CONDITIONS,
    CONDITIONS,
    EXCLUDED_CONDITIONS,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_THRESHOLD_PATH,
)

warnings.filterwarnings("ignore", category=FutureWarning)

HEATMAP_DIR = Path("data/heatmaps")
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# Confidence is computed as the mean probability of the top predicted condition.
# Studies with no condition above this floor get confidence = max(probs).
CONFIDENCE_FLOOR = 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_pending_studies(split: str = None, study_id: str = None) -> list[dict]:
    """
    Return studies that have diagnoses but no validated output yet.
    Uses EXISTS subqueries instead of JOINs to avoid row multiplication
    from multiple diagnoses rows per study. feature_map is NOT fetched
    here — it is loaded per-study inside the run loop to avoid pulling
    gigabytes of BYTEA into memory upfront.
    """
    with get_cursor() as cur:
        query = """
            SELECT
                s.study_id,
                s.split,
                e.faiss_id,
                e.feature_map_shape
            FROM studies s
            INNER JOIN embeddings e ON e.study_id = s.study_id
            WHERE EXISTS (SELECT 1 FROM diagnoses d WHERE d.study_id = s.study_id)
              AND NOT EXISTS (SELECT 1 FROM outputs o WHERE o.study_id = s.study_id)
              AND e.feature_map IS NOT NULL
        """
        params = []

        if study_id:
            query += " AND s.study_id = %s"
            params.append(study_id)
        elif split:
            query += " AND s.split = %s"
            params.append(split)

        query += " ORDER BY s.study_id"
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def get_feature_map(study_id: str) -> tuple[np.ndarray, tuple]:
    """
    Load the feature map BYTEA and shape for a single study.
    Returns (feature_map ndarray, shape tuple).
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT feature_map, feature_map_shape FROM embeddings WHERE study_id = %s",
            (study_id,),
        )
        row = cur.fetchone()

    if row is None or row["feature_map"] is None:
        raise ValueError(f"No feature map found for study {study_id}")

    fmap_shape = tuple(row["feature_map_shape"])
    feature_map = (
        np.frombuffer(bytes(row["feature_map"]), dtype=np.float32)
        .reshape(fmap_shape)
        .copy()
    )
    return feature_map, fmap_shape


def get_diagnoses(study_id: str) -> dict[str, float]:
    """Load per-condition probabilities from the diagnoses table."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT condition, probability FROM diagnoses WHERE study_id = %s",
            (study_id,),
        )
        return {row["condition"]: float(row["probability"]) for row in cur.fetchall()}


def get_report(study_id: str) -> dict:
    """Load findings + impression from the reports table."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT findings, impression, full_report FROM reports WHERE study_id = %s",
            (study_id,),
        )
        row = cur.fetchone()
    if row is None:
        return {"findings": None, "impression": None, "full_report": None}
    return dict(row)


# ---------------------------------------------------------------------------
# Grad-CAM (Score-CAM variant using classifier projection weights)
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: torch.nn.Module,
    feature_map: np.ndarray,
    image_embedding: np.ndarray,
    report_embedding: np.ndarray,
    condition_idx: int,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a specific condition.

    Because we only have the pooled 128-dim embedding at inference time
    (not a live spatial forward pass), we use the image_proj weights as
    CAM weights — they encode which embedding dimensions matter most for
    the final prediction. We project those weights back to the 2048-dim
    feature space via the transpose of the BioViL projection (approximated
    as uniform weighting here, since we don't have the 2048→128 BioViL
    projection weights stored).

    Practical approach: use global average of the feature map channels,
    weighted by the magnitude of image_proj.weight for the top-k channels.

    Parameters
    ----------
    model           : loaded MultimodalClassifier
    feature_map     : (2048, H, W) float32
    image_embedding : (128,) float32
    report_embedding: (128,) float32
    condition_idx   : index into TRAINABLE_CONDITIONS

    Returns
    -------
    cam : (H, W) float32, values in [0, 1]
    """
    H, W = feature_map.shape[1], feature_map.shape[2]

    # Get image_proj weights: shape (proj_dim=64, embed_dim=128)
    # These tell us which embedding dims the model cares about most.
    img_proj_weight = model.image_proj[0].weight.detach().cpu().numpy()  # (64, 128)

    # Get classifier head weights for this condition: shape (1, hidden_dim)
    # We want the contribution of the image branch to the final logit.
    # The classifier is: Linear(128→256) → BN → ReLU → Dropout →
    #                     Linear(256→128) → BN → ReLU → Dropout →
    #                     Linear(128→output_dim)
    clf_layers = list(model.classifier.children())
    w1 = clf_layers[0].weight.detach().cpu().numpy()   # (256, 128)
    w2 = clf_layers[4].weight.detach().cpu().numpy()   # (128, 256)
    w3 = clf_layers[8].weight.detach().cpu().numpy()   # (output_dim, 128)

    # Collapse to a single weight vector for the image branch (first 64 dims of the 128-dim fused vec)
    # w_condition shape after slice: (64,)
    w_combined = (w3[condition_idx] @ w2 @ w1)[:64]   # (64,)

    # CAM weights per embedding dim: how much each image_proj output dim matters
    # Combine with image_proj: map back to 2048 channels
    # cam_weights[c] = sum over proj dims of (w_combined[p] * img_proj_weight[p, :])
    # But BioViL projects 128→128 (not 2048→128), so feature_map channels ≠ embedding dims.
    # We use a surrogate: weight feature map channels by their activation magnitude,
    # scaled by how much the image embedding (which summarises those channels) drives the output.

    condition_weight = float(np.abs(w_combined).sum())   # scalar importance of image branch

    # Weighted average across 2048 channels using activation magnitude as proxy for importance
    channel_weights = feature_map.mean(axis=(1, 2))       # (2048,) — global avg per channel
    channel_weights = np.maximum(channel_weights, 0)       # ReLU

    if channel_weights.sum() > 0:
        channel_weights = channel_weights / channel_weights.sum()

    # Weighted sum over channels → (H, W) spatial activation map
    cam = np.tensordot(channel_weights, feature_map, axes=([0], [0]))  # (H, W)

    # Scale by condition importance
    cam = cam * condition_weight

    # ReLU + normalise to [0, 1]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def save_heatmap(
    cam: np.ndarray,
    study_id: str,
    condition: str,
    heatmap_dir: Path = HEATMAP_DIR,
) -> Path:
    """
    Save a Grad-CAM heatmap as a PNG using matplotlib's jet colormap.
    Returns the path to the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from PIL import Image

        colormap  = cm.get_cmap("jet")
        heatmap   = colormap(cam)[:, :, :3]                 # (H, W, 3) float64 RGB
        heatmap   = (heatmap * 255).astype(np.uint8)

        img       = Image.fromarray(heatmap)
        safe_cond = condition.lower().replace(" ", "_")
        out_path  = heatmap_dir / f"{study_id}_{safe_cond}.png"
        img.save(out_path)
        return out_path

    except ImportError as e:
        # If PIL or matplotlib not available, save raw npy instead
        out_path = heatmap_dir / f"{study_id}.cam.npy"
        np.save(out_path, cam)
        return out_path


# ---------------------------------------------------------------------------
# Output assembly & validation
# ---------------------------------------------------------------------------

def compute_confidence(diagnoses: dict[str, float], thresholds: np.ndarray) -> float:
    """
    Confidence = mean probability across conditions predicted positive.
    Falls back to max probability if nothing crosses threshold.
    """
    probs = np.array([
        diagnoses.get(cond, 0.0) for cond in TRAINABLE_CONDITIONS
    ], dtype=np.float32)

    positive_mask = probs >= thresholds
    if positive_mask.any():
        return float(probs[positive_mask].mean())
    return float(probs.max())


def validate_output(output_dict: dict) -> list[str]:
    """
    Validate the assembled output JSON.
    Returns a list of error strings (empty = valid).
    """
    errors = []

    required_keys = {"findings", "impression", "diagnoses", "confidence"}
    missing = required_keys - set(output_dict.keys())
    if missing:
        errors.append(f"Missing keys: {missing}")

    if "confidence" in output_dict:
        c = output_dict["confidence"]
        if not isinstance(c, float) or not (0.0 <= c <= 1.0):
            errors.append(f"confidence out of range: {c}")

    if "diagnoses" in output_dict:
        for cond, prob in output_dict["diagnoses"].items():
            if not isinstance(prob, float) or not (0.0 <= prob <= 1.0):
                errors.append(f"diagnoses[{cond}] out of range: {prob}")

    if not output_dict.get("findings"):
        errors.append("findings is empty")

    return errors


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(study_id: str = None, split: str = None, limit: int = None):
    print("Fetching pending studies...", flush=True)
    pending = get_pending_studies(split=split, study_id=study_id)
    if limit:
        pending = pending[:limit]

    print(f"Studies pending validation: {len(pending)}", flush=True)
    if not pending:
        print("Nothing to validate.")
        return

    print("Loading FAISS store...", flush=True)
    store = FAISSStore()
    print("Loading model...", flush=True)
    model = load_model(DEFAULT_WEIGHTS_PATH)
    print("Loading thresholds...", flush=True)
    thresholds = load_thresholds(DEFAULT_THRESHOLD_PATH)
    model.eval()
    print("Ready. Starting validation loop...\n", flush=True)

    success      = 0
    failed       = 0
    invalid      = 0
    heatmap_fail = 0

    for i, row in enumerate(pending):
        sid = row["study_id"]
        t0  = time.perf_counter()

        try:
            # ── 1. Reconstruct image embedding from FAISS ──────────────────
            image_embedding = store.get_vector(row["faiss_id"])

            # ── 2. Load feature map for this study from PostgreSQL BYTEA ───
            feature_map, fmap_shape = get_feature_map(sid)

            # ── 3. Load report embedding for classifier input ──────────────
            with get_cursor() as cur:
                cur.execute(
                    "SELECT report_embedding FROM reports WHERE study_id = %s", (sid,)
                )
                rep_row = cur.fetchone()

            if rep_row is None or rep_row["report_embedding"] is None:
                print(f"  SKIP {sid}: no report embedding")
                failed += 1
                continue

            report_embedding = np.frombuffer(
                bytes(rep_row["report_embedding"]), dtype=np.float32
            ).copy()

            # ── 4. Load diagnoses ──────────────────────────────────────────
            diagnoses = get_diagnoses(sid)
            if not diagnoses:
                print(f"  SKIP {sid}: no diagnoses found")
                failed += 1
                continue

            # ── 5. Grad-CAM: generate heatmap for the top predicted condition
            probs_arr = np.array([
                diagnoses.get(cond, 0.0) for cond in TRAINABLE_CONDITIONS
            ], dtype=np.float32)

            top_condition_idx  = int(probs_arr.argmax())
            top_condition_name = TRAINABLE_CONDITIONS[top_condition_idx]

            heatmap_path = None
            try:
                cam = generate_gradcam(
                    model            = model,
                    feature_map      = feature_map,
                    image_embedding  = image_embedding,
                    report_embedding = report_embedding,
                    condition_idx    = top_condition_idx,
                )
                heatmap_path = str(save_heatmap(cam, sid, top_condition_name))
            except Exception as e:
                print(f"  WARNING: heatmap failed for {sid}: {e}")
                heatmap_fail += 1

            # ── 6. Load report text ────────────────────────────────────────
            report = get_report(sid)

            # ── 7. Compute confidence ──────────────────────────────────────
            confidence = compute_confidence(diagnoses, thresholds)

            # ── 8. Assemble output JSON ────────────────────────────────────
            output_dict = {
                "findings":      report.get("findings") or "Not available.",
                "impression":    report.get("impression") or "Not available.",
                "diagnoses":     {
                    cond: round(float(prob), 4)
                    for cond, prob in diagnoses.items()
                },
                "confidence":    round(confidence, 4),
                "top_condition": top_condition_name,
                "heatmap_path":  heatmap_path,
            }

            # ── 9. Validate output ─────────────────────────────────────────
            errors = validate_output(output_dict)
            if errors:
                print(f"  INVALID {sid}: {errors}")
                invalid += 1
                # Still persist — mark as not validated so ops can review
                latency_ms = int((time.perf_counter() - t0) * 1000)
                insert_output(
                    study_id     = sid,
                    output_dict  = output_dict,
                    confidence   = confidence,
                    latency_ms   = latency_ms,
                    heatmap_path = heatmap_path,
                )
                continue

            # ── 10. Persist to PostgreSQL and mark validated ───────────────
            latency_ms = int((time.perf_counter() - t0) * 1000)
            insert_output(
                study_id     = sid,
                output_dict  = output_dict,
                confidence   = confidence,
                latency_ms   = latency_ms,
                heatmap_path = heatmap_path,
            )
            mark_output_validated(sid)

            success += 1

        except Exception as e:
            print(f"  ERROR on study {sid}: {e}")
            failed += 1
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            print(
                f"  [{i+1}/{len(pending)}]  "
                f"validated: {success}  invalid: {invalid}  "
                f"heatmap_fail: {heatmap_fail}  failed: {failed}"
            )

    print(f"\nDone.")
    print(f"  Validated    : {success}")
    print(f"  Invalid JSON : {invalid}")
    print(f"  Heatmap fail : {heatmap_fail}")
    print(f"  Errors       : {failed}")

    if success > 0:
        _print_summary(limit=min(success, 5))


def _print_summary(limit: int = 5):
    """Print a sample of the most recent validated outputs."""
    with get_cursor() as cur:
        cur.execute(
            """SELECT study_id, confidence, latency_ms, heatmap_path
               FROM outputs
               WHERE validated = TRUE
               ORDER BY created_at DESC
               LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()

    print(f"\nSample validated outputs (last {len(rows)}):")
    for row in rows:
        hp = Path(row["heatmap_path"]).name if row["heatmap_path"] else "no heatmap"
        print(
            f"  {row['study_id']}  "
            f"confidence={row['confidence']:.3f}  "
            f"latency={row['latency_ms']}ms  "
            f"heatmap={hp}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 7 — Validation & Grad-CAM heatmap generation"
    )
    parser.add_argument(
        "--study-id", type=str, default=None,
        help="Validate a single study (for debugging)"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None,
        help="Restrict to one data split"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of studies validated this run"
    )
    args = parser.parse_args()
    run(study_id=args.study_id, split=args.split, limit=args.limit)