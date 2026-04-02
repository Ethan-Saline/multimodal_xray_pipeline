"""
infer_new_dicom.py — Single-shot inference on a new DICOM
scripts/ | CheXpert Pipeline

Accepts a DICOM file (or directory of DICOMs for one study) that has
NO existing database entry and runs the full inference pipeline in one
shot, entirely in memory — no database writes unless --save-to-db is
passed.

Pipeline
--------
1.  Encode    — BioViL image encoder → 128-dim embedding + feature map
2.  Retrieve  — FAISS nearest-neighbour search → top-K similar studies
3.  Report    — BioViL-T text model → synthesised findings + 128-dim report embedding
4.  Diagnose  — MultimodalClassifier → per-condition probability scores
5.  Heatmap   — channel-mean Grad-CAM derived from ResNet layer4 feature map
6.  Display   — 2×2 clinical dashboard (raw DICOM | CAM overlay |
                diagnosis bars | radiology report)

Usage
-----
    python -m scripts.infer_new_dicom --dicom path/to/file.dcm
    python -m scripts.infer_new_dicom --dicom path/to/study_dir/
    python -m scripts.infer_new_dicom --dicom study.dcm --save viewer_out.png
    python -m scripts.infer_new_dicom --dicom study.dcm --save-to-db --study-id NEW001
    python -m scripts.infer_new_dicom --dicom study.dcm --top-k 3 --no-display

Options
-------
    --dicom        Path to a .dcm file or a directory containing .dcm files (required)
    --study-id     Temporary study ID used for labels/caching  [default: "new_study"]
    --top-k        Number of similar cases to retrieve         [default: 5]
    --save         Save the dashboard PNG to this path
    --save-to-db   Persist embeddings, report, and diagnoses to PostgreSQL/FAISS
    --no-display   Skip plt.show() — useful for headless environments
"""
# My example to share
# python -m scripts.infer_new_dicom --dicom "C:\Users\ethan\Downloads\000068.dcm"

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

# ── optional pydicom ─────────────────────────────────────────────────────────
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ── pipeline imports ──────────────────────────────────────────────────────────
from health_multimodal.image.utils import ImageModelType, get_image_inference
from health_multimodal.text.utils import get_biovil_t_bert

from models.classifier import load_model, predict, DEFAULT_WEIGHTS_PATH, TRAINABLE_CONDITIONS
from storage.db import (
    get_cursor,
    insert_embedding,
    insert_report,
    insert_diagnosis,
)
from storage.faiss_store import FAISSStore
from storage.cache import cache_embedding
from pipeline.retrieve import retrieve as faiss_retrieve

matplotlib.rcParams["font.family"] = "monospace"

# ── colour palette (mirrors study_viewer.py) ─────────────────────────────────
BG       = "#0d0f14"
PANEL    = "#13161d"
BORDER   = "#1f2433"
TEXT     = "#c8d0e0"
DIM      = "#505878"
ACCENT   = "#3a9bd5"
POSITIVE = "#4fc97e"
WARN     = "#e8a44a"
WHITE    = "#f0f4ff"
CAM_CMAP = "inferno"

DEFAULT_K = 5


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Encode
# ─────────────────────────────────────────────────────────────────────────────

def _get_layer4_module(engine):
    """Return BioViL's ResNet layer4 for the forward hook."""
    try:
        return engine.model.encoder.encoder.layer4
    except AttributeError:
        for name, module in engine.model.named_modules():
            if name.endswith("encoder.layer4"):
                return module
        raise AttributeError("Could not locate layer4 in BioViL model hierarchy.")


def encode_dicom(engine, dicom_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a single DICOM file.

    Returns
    -------
    embedding  : (128,)  float32 projected global embedding
    feature_map: (C, H, W) float32 ResNet layer4 activation map
    """
    captured: dict = {}

    def _hook(module, input, output):
        captured["feature_map"] = output.detach().cpu()

    target_layer = _get_layer4_module(engine)
    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            embedding = engine.get_projected_global_embedding(dicom_path)
        if "feature_map" not in captured:
            raise RuntimeError("Forward hook did not capture feature_map.")
    finally:
        handle.remove()

    emb_np = embedding.cpu().numpy().astype(np.float32)
    fmap   = captured["feature_map"].numpy().astype(np.float32)
    if fmap.ndim == 4:
        fmap = fmap.squeeze(0)          # (1,C,H,W) → (C,H,W)
    return emb_np, fmap


def encode_study(
    engine,
    dicom_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode all DICOMs for a study and return their mean embedding
    and mean feature map (multi-view study support).
    """
    embeddings, fmaps = [], []
    for p in dicom_paths:
        emb, fmap = encode_dicom(engine, p)
        embeddings.append(emb)
        fmaps.append(fmap)
    mean_emb  = np.mean(embeddings, axis=0).astype(np.float32)
    mean_fmap = np.mean(fmaps,      axis=0).astype(np.float32)
    return mean_emb, mean_fmap


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Report (adapted from report.py for in-memory use)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    text_model: tuple,
    image_embedding: np.ndarray,
    retrieved_cases: list[dict],
) -> dict:
    """
    Synthesise a radiology report from retrieved cases and encode it
    into a 128-dim BioViL-T projected text embedding.

    Returns
    -------
    {
        "findings"        : str,
        "impression"      : str,
        "full_report"     : str,
        "report_embedding": np.ndarray (128,)
    }
    """
    tokenizer, model = text_model

    positive_conditions: set[str] = set()
    report_snippets: list[str]    = []

    for case in retrieved_cases:
        for cond, val in case.get("labels", {}).items():
            if val == 1:
                positive_conditions.add(cond)
        if case.get("report_txt"):
            snippet = case["report_txt"][:200].replace("\n", " ").strip()
            report_snippets.append(snippet)

    conditions_str = (
        ", ".join(sorted(positive_conditions)) if positive_conditions else "No Finding"
    )
    findings    = (
        f"Findings consistent with retrieved similar cases. "
        f"Conditions noted: {conditions_str}."
    )
    full_report = f"FINDINGS: {findings}"

    with torch.no_grad():
        inputs = tokenizer(
            full_report,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs          = model(**inputs, output_cls_projected_embedding=True)
        report_embedding = outputs.cls_projected_embedding
        report_embedding = torch.nn.functional.normalize(report_embedding, dim=-1)
        report_embedding = report_embedding.squeeze(0).cpu().numpy().astype(np.float32)

    return {
        "findings":         findings,
        "full_report":      full_report,
        "report_embedding": report_embedding,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — CAM heatmap
# ─────────────────────────────────────────────────────────────────────────────

def derive_cam(feature_map: np.ndarray) -> np.ndarray:
    """
    Derive a normalised channel-mean CAM from a (C, H, W) feature map.
    Returns a (H, W) float32 array in [0, 1].
    """
    cam = np.maximum(feature_map.mean(axis=0), 0).astype(np.float32)
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def resize_cam(cam: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# DICOM pixel loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dicom_pixels(dicom_path: Path) -> np.ndarray | None:
    if not HAS_PYDICOM:
        print("  [viewer] pydicom not installed — raw image unavailable.")
        return None
    try:
        ds  = pydicom.dcmread(str(dicom_path))
        arr = ds.pixel_array.astype(np.float32)
        # Window-level normalisation
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__iter__") \
                 else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth)  if not hasattr(ds.WindowWidth,  "__iter__") \
                 else float(ds.WindowWidth[0])
            arr = np.clip(arr, wc - ww / 2, wc + ww / 2)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255
        return arr.astype(np.uint8)
    except Exception as e:
        print(f"  [viewer] Could not read DICOM pixels: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers  (mirrors study_viewer.py style)
# ─────────────────────────────────────────────────────────────────────────────

def _ax_style(ax, title: str = ""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, color=DIM, fontsize=7.5, loc="left",
                     pad=6, fontweight="bold")


def draw_raw_dicom(ax, pixels: np.ndarray | None):
    _ax_style(ax, "DICOM  ·  raw")
    ax.set_xticks([])
    ax.set_yticks([])
    if pixels is None:
        ax.text(0.5, 0.5, "DICOM unavailable\n(pydicom missing or unreadable)",
                ha="center", va="center", color=DIM, fontsize=8,
                transform=ax.transAxes)
        return
    ax.imshow(pixels, cmap="gray", aspect="equal", interpolation="lanczos")


def draw_cam_overlay(ax, pixels: np.ndarray | None, cam: np.ndarray | None):
    _ax_style(ax, "DICOM  ·  grad-cam overlay")
    ax.set_xticks([])
    ax.set_yticks([])

    if pixels is None and cam is None:
        ax.text(0.5, 0.5, "No image / heatmap data",
                ha="center", va="center", color=DIM, fontsize=8,
                transform=ax.transAxes)
        return

    if pixels is not None:
        ax.imshow(pixels, cmap="gray", aspect="equal", interpolation="lanczos")

    if cam is not None and pixels is not None:
        cam_r = resize_cam(cam, pixels.shape[0], pixels.shape[1])
        ax.imshow(cam_r, cmap=CAM_CMAP, alpha=0.45,
                  aspect="equal", interpolation="lanczos",
                  vmin=0, vmax=1)
    elif cam is not None:
        ax.imshow(cam, cmap=CAM_CMAP, aspect="auto", interpolation="lanczos")


def draw_diagnosis_bars(ax, diagnoses: dict, study_id: str):
    _ax_style(ax, "DIAGNOSIS  ·  model probabilities")

    if not diagnoses:
        ax.text(0.5, 0.5, "No diagnosis data", ha="center", va="center",
                color=DIM, fontsize=8, transform=ax.transAxes)
        return

    # Load thresholds if available
    thresholds = None
    try:
        from models.classifier import load_thresholds, DEFAULT_THRESHOLD_PATH
        thresholds = load_thresholds(DEFAULT_THRESHOLD_PATH)
    except Exception:
        pass

    thresh_map: dict[str, float] = {}
    if thresholds is not None:
        thresh_map = {c: float(thresholds[i]) for i, c in enumerate(TRAINABLE_CONDITIONS)}

    # Sort by probability descending
    items = sorted(diagnoses.items(), key=lambda x: x[1], reverse=True)
    conds = [c for c, _ in items]
    probs = [p for _, p in items]
    n     = len(conds)

    for idx, (cond, prob) in enumerate(zip(conds, probs)):
        thresh  = thresh_map.get(cond, 0.5)
        is_pos  = prob >= thresh
        colour  = POSITIVE if is_pos else (ACCENT if prob > 0.25 else DIM)
        alpha   = 1.0 if is_pos else 0.55

        ax.barh(idx, 1.0,  color=BORDER, height=0.62, zorder=1)
        ax.barh(idx, prob,  color=colour, height=0.62, alpha=alpha, zorder=2)
        ax.axvline(thresh, color=WARN, linewidth=0.6, alpha=0.6, zorder=3)

        axes_y = 1.0 - (idx + 0.5) / n
        label  = f"{'▲ ' if is_pos else '  '}{cond}"
        ax.text(-0.02, axes_y, label,
                ha="right", va="center",
                color=WHITE if is_pos else TEXT,
                fontsize=7.5,
                fontweight="bold" if is_pos else "normal",
                transform=ax.transAxes, clip_on=False)

        ax.text(prob + 0.015, idx, f"{prob:.1%}",
                ha="left", va="center",
                color=colour, fontsize=7,
                fontweight="bold" if is_pos else "normal")

    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       color=DIM, fontsize=6.5)
    ax.tick_params(axis="x", length=3, color=BORDER)
    ax.invert_yaxis()

    if thresh_map and conds:
        ax.text(thresh_map.get(conds[0], 0.5), n - 0.1,
                "threshold", color=WARN, fontsize=6,
                ha="center", va="bottom", style="italic")


def draw_report_panel(ax, study_id: str, report: dict, elapsed_ms: float,
                      diagnoses: dict):
    _ax_style(ax, "RADIOLOGY REPORT  ·  generated")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    findings = report.get("findings", "Not available.")

    # Top-3 diagnoses for the meta block
    top_dx = sorted(diagnoses.items(), key=lambda x: -x[1])[:3]
    top_dx_str = "  ".join(f"{c} {p:.0%}" for c, p in top_dx) if top_dx else "—"

    meta_lines = [
        f"STUDY ID    {study_id}",
        f"SOURCE      NEW DICOM  (not in DB)",
        f"LATENCY     {elapsed_ms:.0f} ms",
        f"TOP DX      {top_dx_str}",
        f"STATUS      ⚡ INFERENCE ONLY",
    ]

    y = 0.97
    for line in meta_lines:
        ax.text(0.03, y, line, transform=ax.transAxes,
                color=DIM, fontsize=7.2, va="top", fontfamily="monospace")
        y -= 0.055

    y -= 0.01
    ax.plot([0.02, 0.98], [y, y], color=BORDER, linewidth=0.8,
            transform=ax.transAxes, clip_on=False)
    y -= 0.035

    def wrap(text: str, max_chars: int = 62) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    ax.text(0.03, y, "FINDINGS", transform=ax.transAxes,
            color=ACCENT, fontsize=7.5, va="top", fontweight="bold",
            fontfamily="monospace")
    y -= 0.045
    for line in wrap(findings)[:10]:
        ax.text(0.03, y, line, transform=ax.transAxes,
                color=TEXT, fontsize=7.2, va="top", fontfamily="monospace")
        y -= 0.042


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard builder
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(
    study_id:   str,
    pixels:     np.ndarray | None,
    cam:        np.ndarray,
    diagnoses:  dict,
    report:     dict,
    elapsed_ms: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)

    fig.text(
        0.5, 0.975,
        f"CHEXPERT PIPELINE  ·  NEW DICOM INFERENCE  ·  {study_id.upper()}",
        ha="center", va="top", color=WHITE, fontsize=11,
        fontweight="bold", fontfamily="monospace",
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.18, right=0.98, top=0.94, bottom=0.04,
        hspace=0.14, wspace=0.08,
        height_ratios=[1.4, 1],
        width_ratios=[1, 1],
    )

    ax_raw    = fig.add_subplot(gs[0, 0])
    ax_cam    = fig.add_subplot(gs[0, 1])
    ax_diag   = fig.add_subplot(gs[1, 0])
    ax_report = fig.add_subplot(gs[1, 1])

    for ax in (ax_raw, ax_cam, ax_diag, ax_report):
        ax.set_facecolor(PANEL)

    draw_raw_dicom(ax_raw, pixels)
    draw_cam_overlay(ax_cam, pixels, cam)
    draw_diagnosis_bars(ax_diag, diagnoses, study_id)
    draw_report_panel(ax_report, study_id, report, elapsed_ms, diagnoses)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Optional DB persist
# ─────────────────────────────────────────────────────────────────────────────

def persist_to_db(
    study_id:        str,
    embedding:       np.ndarray,
    feature_map:     np.ndarray,
    report:          dict,
    diagnoses:       dict,
):
    """
    Write results to PostgreSQL + FAISS.
    Called only when --save-to-db is set.
    Requires a studies row to already exist (or inserts a minimal one).
    """
    store    = FAISSStore()
    faiss_id = store.add(embedding)

    insert_embedding(
        study_id          = study_id,
        dicom_uid         = f"s{study_id}",
        faiss_id          = faiss_id,
        model_name        = "biovil",
        embed_dim         = embedding.shape[0],
        feature_map       = feature_map.tobytes(),
        feature_map_shape = list(feature_map.shape),
    )
    cache_embedding(f"img_{study_id}", embedding)

    insert_report(
        study_id         = study_id,
        findings         = report["findings"],
        full_report      = report["full_report"],
        report_embedding = report["report_embedding"],
    )

    insert_diagnosis(study_id, diagnoses)
    print(f"  Saved to DB — FAISS id: {faiss_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    dicom_input:  Path,
    study_id:     str  = "new_study",
    top_k:        int  = DEFAULT_K,
    save_path:    Path = None,
    save_to_db:   bool = False,
    no_display:   bool = False,
):
    t_start = time.perf_counter()

    # ── Resolve DICOM paths ───────────────────────────────────────────────────
    if dicom_input.is_dir():
        dicom_paths = sorted(dicom_input.glob("*.dcm"))
        if not dicom_paths:
            sys.exit(f"ERROR: No .dcm files found in {dicom_input}")
        print(f"Found {len(dicom_paths)} DICOM(s) in {dicom_input}")
    elif dicom_input.is_file():
        dicom_paths = [dicom_input]
    else:
        sys.exit(f"ERROR: {dicom_input} does not exist.")

    primary_dicom = dicom_paths[0]

    # ── Step 1: Encode ────────────────────────────────────────────────────────
    print("\n[1/4] Encoding DICOM(s) with BioViL...")
    engine = get_image_inference(ImageModelType.BIOVIL)
    embedding, feature_map = encode_study(engine, dicom_paths)
    print(f"  Embedding shape  : {embedding.shape}")
    print(f"  Feature map shape: {feature_map.shape}")

    # ── Step 2: Retrieve ──────────────────────────────────────────────────────
    print(f"\n[2/4] Retrieving top-{top_k} similar cases from FAISS...")
    retrieved = faiss_retrieve(embedding, k=top_k)
    print(f"  Retrieved {len(retrieved)} case(s)")
    for i, case in enumerate(retrieved, 1):
        pos = [c for c, v in case.get("labels", {}).items() if v == 1]
        print(f"    {i}. {case['study_id']}  dist={case['distance']:.4f}"
              f"  labels={', '.join(pos) or 'none'}")

    # ── Step 3: Report ────────────────────────────────────────────────────────
    print("\n[3/4] Generating radiology report with BioViL-T...")
    tokenizer, text_model_obj = get_biovil_t_bert()
    text_model_obj.eval()
    report = generate_report((tokenizer, text_model_obj), embedding, retrieved)
    print(f"  Findings   : {report['findings'][:80]}...")

    # ── Step 4: Diagnose ──────────────────────────────────────────────────────
    print("\n[4/4] Running MultimodalClassifier...")
    if not DEFAULT_WEIGHTS_PATH.exists():
        sys.exit(
            f"ERROR: No classifier weights at {DEFAULT_WEIGHTS_PATH}.\n"
            f"Run train_classifier.py first."
        )
    clf_model = load_model(DEFAULT_WEIGHTS_PATH)
    clf_model.eval()
    diagnoses = predict(clf_model, embedding, report["report_embedding"])
    top5 = sorted(diagnoses.items(), key=lambda x: -x[1])[:5]
    print("  Top predictions:")
    for cond, prob in top5:
        bar = "█" * int(prob * 20)
        print(f"    {cond:<32} {bar:<20} {prob:.1%}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    print(f"\nTotal inference time: {elapsed_ms:.0f} ms")

    # ── Step 5: Derive CAM heatmap ────────────────────────────────────────────
    cam    = derive_cam(feature_map)
    pixels = load_dicom_pixels(primary_dicom)

    # ── Step 6: Render dashboard ──────────────────────────────────────────────
    print("\nRendering clinical dashboard...")
    fig = build_figure(
        study_id    = study_id,
        pixels      = pixels,
        cam         = cam,
        diagnoses   = diagnoses,
        report      = report,
        elapsed_ms  = elapsed_ms,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Dashboard saved → {save_path}")

    if not no_display:
        plt.show()

    plt.close(fig)

    # ── Optional: persist to DB ───────────────────────────────────────────────
    if save_to_db:
        print("\nPersisting to database...")
        persist_to_db(
            study_id    = study_id,
            embedding   = embedding,
            feature_map = feature_map,
            report      = report,
            diagnoses   = diagnoses,
        )

    print("\nDone.")
    return {
        "study_id":         study_id,
        "embedding":        embedding,
        "feature_map":      feature_map,
        "report":           report,
        "diagnoses":        diagnoses,
        "cam":              cam,
        "elapsed_ms":       elapsed_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-shot inference on a new DICOM — no prior DB entry required."
    )
    parser.add_argument(
        "--dicom", type=Path, required=True,
        help="Path to a .dcm file or a directory of .dcm files for one study.",
    )
    parser.add_argument(
        "--study-id", type=str, default="new_study",
        help="Temporary label used for display and optional DB persist. "
             "[default: 'new_study']",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_K,
        help=f"Number of similar cases to retrieve from FAISS. [default: {DEFAULT_K}]",
    )
    parser.add_argument(
        "--save", type=Path, default=None, metavar="PATH",
        help="Save the dashboard PNG to this path instead of (or in addition to) "
             "displaying it.",
    )
    parser.add_argument(
        "--save-to-db", action="store_true",
        help="Persist embeddings, report, and diagnoses to PostgreSQL + FAISS.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Skip plt.show() — useful in headless / batch environments.",
    )

    args = parser.parse_args()
    run(
        dicom_input = args.dicom,
        study_id    = args.study_id,
        top_k       = args.top_k,
        save_path   = args.save,
        save_to_db  = args.save_to_db,
        no_display  = args.no_display,
    )