"""
train_classifier.py — Weekly CNN retraining (v2)
scripts/ | CheXpert Pipeline

Loads train + val splits, trains with:
  - Focal loss
  - Validation AUROC after every epoch
  - Early stopping
  - Per-condition threshold tuning on val set

Usage:
    python -m scripts.train_classifier
    python -m scripts.train_classifier --epochs 100
    python -m scripts.train_classifier --patience 15
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch

from models.classifier import (
    train, compute_auroc, load_model,
    TRAINABLE_CONDITIONS,
    DEFAULT_WEIGHTS_PATH, DEFAULT_THRESHOLD_PATH,
)
from storage.db import get_cursor
from storage.faiss_store import FAISSStore


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #

def load_split(split: str, store: FAISSStore) -> tuple:
    """
    Load image embeddings, report embeddings, and labels for one split.
    Returns (image_embeddings, report_embeddings, labels, study_ids).
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT s.study_id, e.faiss_id, r.report_embedding
            FROM   studies s
            INNER JOIN embeddings e ON s.study_id = e.study_id
            INNER JOIN reports    r ON s.study_id = r.study_id
            WHERE  r.report_embedding IS NOT NULL
              AND  s.split = %s
            ORDER  BY s.study_id
        """, (split,))
        rows = cur.fetchall()

        if not rows:
            return None, None, None, []

        study_ids = [row["study_id"] for row in rows]

        cur.execute("""
            SELECT study_id, condition, mapped_value
            FROM   label_mappings
            WHERE  study_id = ANY(%s)
        """, (study_ids,))
        label_rows = cur.fetchall()

    labels_by_study = defaultdict(dict)
    for lr in label_rows:
        labels_by_study[lr["study_id"]][lr["condition"]] = lr["mapped_value"]

    image_embs  = []
    report_embs = []
    labels_list = []
    valid_ids   = []
    skipped     = 0

    for row in rows:
        sid = row["study_id"]

        try:
            img_emb = store.get_vector(row["faiss_id"])
        except Exception:
            skipped += 1
            continue

        raw = row["report_embedding"]
        if raw is None:
            skipped += 1
            continue
        rep_emb = np.frombuffer(bytes(raw), dtype=np.float32).copy()
        if rep_emb.shape[0] != 128:
            skipped += 1
            continue

        label_vec = np.array([
            labels_by_study[sid].get(cond, 0)
            for cond in TRAINABLE_CONDITIONS
        ], dtype=np.float32)

        image_embs.append(img_emb)
        report_embs.append(rep_emb)
        labels_list.append(label_vec)
        valid_ids.append(sid)

    if skipped:
        print(f"    Skipped {skipped} studies (missing embedding)")

    if not image_embs:
        return None, None, None, []

    return (
        np.stack(image_embs),
        np.stack(report_embs),
        np.stack(labels_list),
        valid_ids,
    )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(epochs: int = 100, early_stopping_patience: int = 10):
    store = FAISSStore()
    print(f"FAISS index size: {store.size}")

    print("\nLoading train split...")
    tr_img, tr_rep, tr_lbl, tr_ids = load_split("train", store)
    print(f"  Train studies : {len(tr_ids) if tr_ids else 0}")

    print("Loading val split...")
    va_img, va_rep, va_lbl, va_ids = load_split("val", store)
    print(f"  Val studies   : {len(va_ids) if va_ids else 0}")

    if tr_img is None or len(tr_ids) == 0:
        print("No training data found. Make sure encode.py and report.py have run.")
        return

    print(f"\nLabel distribution (train):")
    for i, cond in enumerate(TRAINABLE_CONDITIONS):
        pos = int(tr_lbl[:, i].sum())
        pct = 100 * pos / len(tr_lbl)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {cond:<30} {bar} {pos:>4} ({pct:.1f}%)")

    print(f"\nTraining for up to {epochs} epochs "
          f"(early stopping patience={early_stopping_patience})...")

    result = train(
        image_embeddings        = tr_img,
        report_embeddings       = tr_rep,
        labels                  = tr_lbl,
        val_image_embeddings    = va_img,
        val_report_embeddings   = va_rep,
        val_labels              = va_lbl,
        epochs                  = epochs,
        early_stopping_patience = early_stopping_patience,
        weights_path            = DEFAULT_WEIGHTS_PATH,
        threshold_path          = DEFAULT_THRESHOLD_PATH,
    )

    print(f"\n{'='*50}")
    print(f"Training complete — {result['epochs_run']} epochs run")
    print(f"  Final train loss : {result['final_loss']:.4f}")
    print(f"  Best val AUROC   : {result['best_val_auroc']:.4f}")

    # Final evaluation on test set
    print(f"\nLoading test split for final evaluation...")
    te_img, te_rep, te_lbl, te_ids = load_split("test", store)
    print(f"  Test studies : {len(te_ids) if te_ids else 0}")

    if te_img is not None and len(te_ids) > 0:
        model = load_model(DEFAULT_WEIGHTS_PATH)
        with torch.no_grad():
            t_img  = torch.tensor(te_img, dtype=torch.float32)
            t_rep  = torch.tensor(te_rep, dtype=torch.float32)
            tprobs = torch.sigmoid(model(t_img, t_rep)).numpy()

        aurocs = compute_auroc(te_lbl, tprobs)
        valid  = {k: v for k, v in aurocs.items() if not np.isnan(v)}
        mean   = np.mean(list(valid.values())) if valid else 0.0

        print(f"\nTest AUROC per condition:")
        for cond in TRAINABLE_CONDITIONS:
            auc = aurocs.get(cond, float("nan"))
            if np.isnan(auc):
                line = "  (insufficient positives)"
            else:
                bar  = "█" * int(auc * 20) + "░" * (20 - int(auc * 20))
                flag = " ✓" if auc >= 0.85 else (" ~" if auc >= 0.75 else " ✗")
                line = f"  {bar}  {auc:.3f}{flag}"
            print(f"  {cond:<30}{line}")

        print(f"\n  Mean test AUROC : {mean:.4f}")
        print(f"  Target          : > 0.850")
        print(f"  {'✓ TARGET MET' if mean >= 0.85 else '✗ BELOW TARGET — more data needed'}")

    # Loss trend
    losses = result["train_losses"]
    if len(losses) >= 10:
        early_avg = np.mean(losses[:5])
        late_avg  = np.mean(losses[-5:])
        drop      = 100 * (early_avg - late_avg) / early_avg
        print(f"\n  Loss trend : {early_avg:.4f} → {late_avg:.4f}  ({drop:.1f}% improvement)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=100,
                        help="Max epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    args = parser.parse_args()
    run(epochs=args.epochs, early_stopping_patience=args.patience)