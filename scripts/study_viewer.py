"""
study_viewer.py — Clinical Study Viewer
scripts/ | CheXpert Pipeline

Displays a single study as a clinical dashboard:
  ┌─────────────────────┬─────────────────────┐
  │   DICOM (raw)       │  DICOM + CAM overlay │
  ├─────────────────────┼─────────────────────┤
  │   Diagnosis bars    │  Report text         │
  └─────────────────────┴─────────────────────┘

Usage:
    python -m scripts.study_viewer
    python -m scripts.study_viewer --study-id 50414267
    python -m scripts.study_viewer --save
    python -m scripts.study_viewer --study-id 50414267 --save --out results/viewer.png

Dependencies: pydicom, matplotlib, pillow, numpy, psycopg2
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from storage.db import get_cursor
from models.classifier import TRAINABLE_CONDITIONS, CONDITIONS, EXCLUDED_CONDITIONS

matplotlib.rcParams["font.family"] = "monospace"

# ── colour palette ────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def pick_study_id() -> str:
    with get_cursor() as cur:
        cur.execute(
            """SELECT s.study_id FROM studies s
               INNER JOIN outputs o ON o.study_id = s.study_id
               WHERE o.validated = TRUE
               ORDER BY o.created_at DESC
               LIMIT 1"""
        )
        row = cur.fetchone()
    if row is None:
        sys.exit("No validated studies found. Run validate.py first.")
    return row["study_id"]


def load_study(study_id: str) -> dict:
    with get_cursor() as cur:
        # feature_map intentionally excluded — too large to pull upfront
        cur.execute(
            """SELECT s.study_id, s.subject_id, s.split, s.report_txt,
                      r.findings, r.impression, r.full_report,
                      e.faiss_id, e.feature_map_shape, e.dicom_uid,
                      o.output_json, o.confidence, o.latency_ms, o.heatmap_path,
                      o.validated
               FROM   studies   s
               INNER  JOIN embeddings e ON e.study_id = s.study_id
               LEFT   JOIN reports    r ON r.study_id = s.study_id
               LEFT   JOIN outputs    o ON o.study_id = s.study_id
               WHERE  s.study_id = %s""",
            (study_id,),
        )
        row = cur.fetchone()
        if row is None:
            sys.exit(f"Study '{study_id}' not found in database.")
        data = dict(row)

        cur.execute(
            """SELECT condition, probability FROM diagnoses
               WHERE study_id = %s ORDER BY probability DESC""",
            (study_id,),
        )
        data["diagnoses"] = {
            r["condition"]: float(r["probability"]) for r in cur.fetchall()
        }

    data["dicom_path"] = _find_dicom_path(study_id, data["subject_id"])
    return data


def _find_dicom_path(study_id: str, subject_id: str) -> Path | None:
    manifest = Path("data/raw/manifest.csv")
    if not manifest.exists():
        return None
    import csv
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            if row["study_id"] == study_id:
                p = Path(row["dicom_path"])
                if p.exists():
                    return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dicom_pixels(dicom_path: Path) -> np.ndarray | None:
    if dicom_path is None or not HAS_PYDICOM:
        return None
    try:
        ds  = pydicom.dcmread(str(dicom_path))
        arr = ds.pixel_array.astype(np.float32)

        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__iter__") \
                 else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth) if not hasattr(ds.WindowWidth, "__iter__") \
                 else float(ds.WindowWidth[0])
            arr = np.clip(arr, wc - ww / 2, wc + ww / 2)

        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255
        return arr.astype(np.uint8)

    except Exception as e:
        print(f"  [viewer] Could not read DICOM: {e}")
        return None


def load_cam(data: dict) -> np.ndarray | None:
    """
    Load Grad-CAM. Tries saved PNG/npy first, then derives from feature_map.
    """
    hp = data.get("heatmap_path")
    if hp:
        p = Path(hp)
        if p.suffix == ".png" and p.exists():
            try:
                from PIL import Image
                return np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            except Exception:
                pass
        if p.suffix == ".npy" and p.exists():
            try:
                return np.load(p).astype(np.float32)
            except Exception:
                pass

    # Fallback: derive channel-mean CAM from DB feature_map
    shape = data.get("feature_map_shape")
    if shape is not None:
        try:
            with get_cursor() as cur:
                cur.execute(
                    "SELECT feature_map FROM embeddings WHERE study_id = %s",
                    (data["study_id"],),
                )
                row = cur.fetchone()
            if row and row["feature_map"]:
                fmap = (
                    np.frombuffer(bytes(row["feature_map"]), dtype=np.float32)
                    .reshape(tuple(shape))
                    .copy()
                )
                cam = np.maximum(fmap.mean(axis=0), 0)
                if cam.max() > 0:
                    cam /= cam.max()
                return cam.astype(np.float32)
        except Exception as e:
            print(f"  [viewer] Could not derive CAM from feature_map: {e}")

    return None


def resize_cam_to(cam: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ax_style(ax, title: str = ""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, color=DIM, fontsize=7.5, loc="left",
                     pad=6, fontweight="bold")


def draw_dicom(ax, pixels: np.ndarray | None, title: str):
    _ax_style(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])
    if pixels is None:
        ax.text(0.5, 0.5, "DICOM unavailable\n(file not found or pydicom missing)",
                ha="center", va="center", color=DIM, fontsize=8,
                transform=ax.transAxes)
        return
    ax.imshow(pixels, cmap="gray", aspect="equal", interpolation="lanczos")


def draw_cam_overlay(ax, pixels: np.ndarray | None, cam: np.ndarray | None, title: str):
    _ax_style(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])

    if pixels is None and cam is None:
        ax.text(0.5, 0.5, "No image / heatmap data", ha="center", va="center",
                color=DIM, fontsize=8, transform=ax.transAxes)
        return

    if pixels is not None:
        ax.imshow(pixels, cmap="gray", aspect="equal", interpolation="lanczos")

    if cam is not None:
        h = pixels.shape[0] if pixels is not None else 224
        w = pixels.shape[1] if pixels is not None else 224
        cam_r = resize_cam_to(cam, h, w) if cam.shape != (h, w) else cam
        ax.imshow(cam_r, cmap=CAM_CMAP, alpha=0.50, aspect="equal",
                  vmin=0, vmax=1, interpolation="bilinear")
        sm = plt.cm.ScalarMappable(cmap=CAM_CMAP, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=6, colors=DIM)
        cbar.outline.set_edgecolor(BORDER)
        cbar.set_label("Activation", color=DIM, fontsize=6)
    else:
        ax.text(0.02, 0.02, "heatmap not available", ha="left", va="bottom",
                color=DIM, fontsize=6.5, transform=ax.transAxes, style="italic")


def draw_diagnosis_bars(ax, diagnoses: dict[str, float], thresholds: np.ndarray | None):
    _ax_style(ax, "DIAGNOSIS  ·  probability scores")

    if not diagnoses:
        ax.text(0.5, 0.5, "No diagnoses available", ha="center", va="center",
                color=DIM, fontsize=9, transform=ax.transAxes)
        return

    conds = [c for c in CONDITIONS if c not in EXCLUDED_CONDITIONS]
    probs = [diagnoses.get(c, 0.0) for c in conds]

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    conds = [conds[i] for i in order]
    probs = [probs[i]  for i in order]

    thresh_map = {}
    if thresholds is not None:
        thresh_map = {c: float(thresholds[i]) for i, c in enumerate(TRAINABLE_CONDITIONS)}

    n = len(conds)

    for idx, (cond, prob) in enumerate(zip(conds, probs)):
        thresh  = thresh_map.get(cond, 0.5)
        is_pos  = prob >= thresh
        colour  = POSITIVE if is_pos else (ACCENT if prob > 0.25 else DIM)
        alpha   = 1.0 if is_pos else 0.55

        # Background track
        ax.barh(idx, 1.0, color=BORDER, height=0.62, zorder=1)
        # Value bar
        ax.barh(idx, prob, color=colour, height=0.62, alpha=alpha, zorder=2)
        # Threshold marker
        ax.axvline(thresh, color=WARN, linewidth=0.6, alpha=0.6, zorder=3)

        # Condition label in axes coordinates so it renders outside the plot area
        # without being clipped — (0,0) = bottom-left, (1,1) = top-right of axes
        axes_y = 1.0 - (idx + 0.5) / n   # invert because y-axis is inverted below
        label  = f"{'▲ ' if is_pos else '  '}{cond}"
        ax.text(
            -0.02, axes_y, label,
            ha="right", va="center",
            color=WHITE if is_pos else TEXT,
            fontsize=7.5,
            fontweight="bold" if is_pos else "normal",
            transform=ax.transAxes,
            clip_on=False,
        )

        # Probability value just beyond the bar end (data coords)
        ax.text(prob + 0.015, idx, f"{prob:.1%}", ha="left", va="center",
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


def draw_report(ax, data: dict):
    _ax_style(ax, "RADIOLOGY REPORT")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    findings   = data.get("findings")   or data.get("report_txt") or "Not available."
    impression = data.get("impression") or ""
    confidence = data.get("confidence")
    latency    = data.get("latency_ms")
    validated  = data.get("validated")

    meta_lines = [
        f"STUDY ID    {data['study_id']}",
        f"SUBJECT     {data['subject_id']}",
        f"SPLIT       {data.get('split', '—').upper()}",
    ]
    if confidence is not None:
        bar_n   = int(confidence * 12)
        bar_str = "█" * bar_n + "░" * (12 - bar_n)
        meta_lines.append(f"CONFIDENCE  {bar_str}  {confidence:.1%}")
    if latency is not None:
        meta_lines.append(f"LATENCY     {latency} ms")
    if validated is not None:
        meta_lines.append(f"VALIDATED   {'✓ YES' if validated else '✗ NO'}")

    y = 0.97
    for line in meta_lines:
        ax.text(0.03, y, line, transform=ax.transAxes,
                color=DIM, fontsize=7.2, va="top", fontfamily="monospace")
        y -= 0.055

    y -= 0.01
    # ax.plot instead of ax.axhline — axhline does not accept transform kwarg
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
    for line in wrap(findings)[:8]:
        ax.text(0.03, y, line, transform=ax.transAxes,
                color=TEXT, fontsize=7.2, va="top", fontfamily="monospace")
        y -= 0.042

    y -= 0.02
    if impression:
        ax.text(0.03, y, "IMPRESSION", transform=ax.transAxes,
                color=POSITIVE, fontsize=7.5, va="top", fontweight="bold",
                fontfamily="monospace")
        y -= 0.045
        for line in wrap(impression)[:5]:
            ax.text(0.03, y, line, transform=ax.transAxes,
                    color=TEXT, fontsize=7.2, va="top", fontfamily="monospace")
            y -= 0.042

    hp = data.get("heatmap_path")
    if hp:
        ax.text(0.03, 0.02, f"CAM  {Path(hp).name}",
                transform=ax.transAxes, color=DIM, fontsize=6.5,
                va="bottom", fontfamily="monospace", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(data: dict) -> plt.Figure:
    try:
        from models.classifier import load_thresholds, DEFAULT_THRESHOLD_PATH
        thresholds = load_thresholds(DEFAULT_THRESHOLD_PATH)
    except Exception:
        thresholds = None

    pixels = load_dicom_pixels(data.get("dicom_path"))
    cam    = load_cam(data)

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)

    fig.text(
        0.5, 0.975,
        f"CHEXPERT PIPELINE  ·  STUDY {data['study_id']}  ·  {data.get('split', '').upper()}",
        ha="center", va="top", color=WHITE, fontsize=11,
        fontweight="bold", fontfamily="monospace",
    )

    # left=0.18 gives the diagnosis condition labels room to render fully
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.18,
        right=0.98,
        top=0.94,
        bottom=0.04,
        hspace=0.14,
        wspace=0.08,
        height_ratios=[1.4, 1],
        width_ratios=[1, 1],
    )

    ax_raw    = fig.add_subplot(gs[0, 0])
    ax_cam    = fig.add_subplot(gs[0, 1])
    ax_diag   = fig.add_subplot(gs[1, 0])
    ax_report = fig.add_subplot(gs[1, 1])

    for ax in (ax_raw, ax_cam, ax_diag, ax_report):
        ax.set_facecolor(PANEL)

    draw_dicom(ax_raw, pixels, "DICOM  ·  raw")
    draw_cam_overlay(ax_cam, pixels, cam, "DICOM  ·  grad-cam overlay")
    draw_diagnosis_bars(ax_diag, data["diagnoses"], thresholds)
    draw_report(ax_report, data)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(study_id: str = None, save: bool = False, out: str = None):
    if study_id is None:
        print("No --study-id given, using most recent validated study...")
        study_id = pick_study_id()

    print(f"Loading study {study_id}...")
    data = load_study(study_id)
    print(f"  Subject : {data['subject_id']}")
    print(f"  Split   : {data.get('split')}")
    print(f"  DICOM   : {data.get('dicom_path') or 'not found'}")
    print(f"  CAM     : {data.get('heatmap_path') or 'will derive from feature_map'}")

    print("Building figure...")
    fig = build_figure(data)

    if save or out:
        out_path = Path(out) if out else Path(f"data/viewer_{study_id}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clinical study viewer — DICOM + CAM + report + diagnosis"
    )
    parser.add_argument(
        "--study-id", type=str, default=None,
        help="Study ID to display (default: most recent validated)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save to PNG instead of opening a window"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for --save (default: data/viewer_{study_id}.png)"
    )
    args = parser.parse_args()
    run(study_id=args.study_id, save=args.save, out=args.out)