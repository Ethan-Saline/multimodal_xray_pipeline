"""
classifier.py — Multimodal CNN classifier (v2)
models/ | CheXpert Pipeline

Improvements over v1:
  1. Validation loop with AUROC + early stopping
  2. Focal loss + per-condition threshold tuning
  3. Learnable projection layers for image and report embeddings
  4. Stronger regularization + reproducibility seeds
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ #
# Reproducibility                                                      #
# ------------------------------------------------------------------ #

SEED = 42

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
# Conditions                                                           #
# ------------------------------------------------------------------ #

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

EXCLUDED_CONDITIONS  = {"Fracture", "Pleural Other"}
TRAINABLE_CONDITIONS = [c for c in CONDITIONS if c not in EXCLUDED_CONDITIONS]

EMBED_DIM  = 128   # BioViL / BioViL-T projected embedding dimension
PROJ_DIM   = 64    # learned projection dim per modality
HIDDEN_DIM = 256
OUTPUT_DIM = len(TRAINABLE_CONDITIONS)

DEFAULT_WEIGHTS_PATH   = Path("models/classifier.pt")
DEFAULT_THRESHOLD_PATH = Path("models/thresholds.npy")


# ------------------------------------------------------------------ #
# Focal Loss                                                           #
# ------------------------------------------------------------------ #

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal loss for multi-label classification.
    Down-weights easy examples so the model focuses on hard ones.
    gamma=2 is the standard default. pos_weight handles imbalance.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none",
        )
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        return (bce * ((1 - p_t) ** self.gamma)).mean()


# ------------------------------------------------------------------ #
# Model                                                                #
# ------------------------------------------------------------------ #

class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier with learnable projection layers.

    Image and report embeddings are projected through separate learned
    linear layers before fusion. This lets the model learn which
    dimensions of each embedding are most useful for classification,
    rather than treating all 256 concatenated dims equally.

    image_embedding (128)  → image_proj  (64)  ─┐
                                                  ├─ concat (128) → classifier → logits
    report_embedding (128) → report_proj (64) ─┘
    """
    def __init__(self, embed_dim=EMBED_DIM, proj_dim=PROJ_DIM,
                 hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=0.4):
        super().__init__()

        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.report_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        fused_dim = proj_dim * 2  # 128

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_emb: torch.Tensor, report_emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            torch.cat([self.image_proj(image_emb), self.report_proj(report_emb)], dim=-1)
        )


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, image_embeddings, report_embeddings, labels):
        self.image  = torch.tensor(image_embeddings,  dtype=torch.float32)
        self.report = torch.tensor(report_embeddings, dtype=torch.float32)
        self.y      = torch.tensor(labels,            dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.image[idx], self.report[idx], self.y[idx]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    n       = labels.shape[0]
    pos     = labels.sum(axis=0).clip(min=1)
    neg     = (n - labels.sum(axis=0)).clip(min=1)
    return torch.tensor(np.clip(neg / pos, 1.0, 20.0), dtype=torch.float32)


def compute_auroc(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    results = {}
    for i, cond in enumerate(TRAINABLE_CONDITIONS):
        col = labels[:, i]
        if col.sum() == 0 or (col == 0).all():
            results[cond] = float("nan")
            continue
        try:
            results[cond] = roc_auc_score(col, probs[:, i])
        except Exception:
            results[cond] = float("nan")
    return results


def tune_thresholds(labels: np.ndarray, probs: np.ndarray,
                    threshold_path: Path = DEFAULT_THRESHOLD_PATH) -> np.ndarray:
    """Find threshold per condition that maximises F1. Saves to disk."""
    from sklearn.metrics import f1_score
    thresholds = np.full(len(TRAINABLE_CONDITIONS), 0.5)
    for i in range(len(TRAINABLE_CONDITIONS)):
        col = labels[:, i]
        if col.sum() == 0:
            continue
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(col, (probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    np.save(threshold_path, thresholds)
    return thresholds


def load_thresholds(threshold_path: Path = DEFAULT_THRESHOLD_PATH) -> np.ndarray:
    if threshold_path.exists():
        return np.load(threshold_path)
    return np.full(len(TRAINABLE_CONDITIONS), 0.5)


# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #

def train(
    image_embeddings: np.ndarray,
    report_embeddings: np.ndarray,
    labels: np.ndarray,
    val_image_embeddings: np.ndarray = None,
    val_report_embeddings: np.ndarray = None,
    val_labels: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
    seed: int = SEED,
) -> dict:

    set_seed(seed)

    dataset = EmbeddingDataset(image_embeddings, report_embeddings, labels)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    has_val = (val_image_embeddings is not None
               and val_labels is not None
               and len(val_image_embeddings) > 0)

    pos_weights = compute_pos_weights(labels)
    model       = MultimodalClassifier()
    criterion   = FocalBCEWithLogitsLoss(gamma=2.0, pos_weight=pos_weights)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_auroc = -1.0
    best_state     = None
    patience_count = 0
    train_losses   = []
    val_aurocs     = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for img, rep, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(img, rep), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                vimg   = torch.tensor(val_image_embeddings,  dtype=torch.float32)
                vrep   = torch.tensor(val_report_embeddings, dtype=torch.float32)
                vprobs = torch.sigmoid(model(vimg, vrep)).numpy()

            aurocs     = compute_auroc(val_labels, vprobs)
            valid_aucs = [v for v in aurocs.values() if not np.isnan(v)]
            mean_auroc = np.mean(valid_aucs) if valid_aucs else 0.0
            val_aurocs.append(mean_auroc)

            if mean_auroc > best_val_auroc:
                best_val_auroc = mean_auroc
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3}/{epochs}  loss={avg_loss:.4f}  "
                      f"val_auroc={mean_auroc:.4f}  best={best_val_auroc:.4f}  "
                      f"patience={patience_count}/{early_stopping_patience}")

            if patience_count >= early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3}/{epochs}  loss={avg_loss:.4f}")

    # Restore best validated state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best model (val_auroc={best_val_auroc:.4f})")

    # Tune thresholds on validation set
    if has_val:
        model.eval()
        with torch.no_grad():
            vimg   = torch.tensor(val_image_embeddings,  dtype=torch.float32)
            vrep   = torch.tensor(val_report_embeddings, dtype=torch.float32)
            vprobs = torch.sigmoid(model(vimg, vrep)).numpy()
        thresholds = tune_thresholds(val_labels, vprobs, threshold_path)
        print(f"  Thresholds tuned → {threshold_path}")
    else:
        thresholds = np.full(len(TRAINABLE_CONDITIONS), 0.5)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "conditions":       TRAINABLE_CONDITIONS,
        "embed_dim":        EMBED_DIM,
        "proj_dim":         PROJ_DIM,
        "hidden_dim":       HIDDEN_DIM,
        "output_dim":       OUTPUT_DIM,
        "thresholds":       thresholds.tolist(),
    }, weights_path)
    print(f"  Weights saved → {weights_path}")

    return {
        "final_loss":     train_losses[-1],
        "best_val_auroc": best_val_auroc,
        "train_losses":   train_losses,
        "val_aurocs":     val_aurocs,
        "epochs_run":     len(train_losses),
    }


# ------------------------------------------------------------------ #
# Inference                                                            #
# ------------------------------------------------------------------ #

def load_model(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> MultimodalClassifier:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No weights at {weights_path}. Run train_classifier.py first."
        )
    ckpt  = torch.load(weights_path, map_location="cpu")
    model = MultimodalClassifier(
        embed_dim  = ckpt.get("embed_dim",  EMBED_DIM),
        proj_dim   = ckpt.get("proj_dim",   PROJ_DIM),
        hidden_dim = ckpt.get("hidden_dim", HIDDEN_DIM),
        output_dim = ckpt.get("output_dim", OUTPUT_DIM),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(
    model: MultimodalClassifier,
    image_embedding: np.ndarray,
    report_embedding: np.ndarray,
) -> dict[str, float]:
    """
    Run inference on one study.
    Returns {condition: probability} for all 14 CONDITIONS.
    Excluded conditions return 0.0.
    """
    img = torch.tensor(image_embedding.astype(np.float32)).unsqueeze(0)
    rep = torch.tensor(report_embedding.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(img, rep)).squeeze(0).numpy()

    result, prob_idx = {}, 0
    for cond in CONDITIONS:
        if cond in EXCLUDED_CONDITIONS:
            result[cond] = 0.0
        else:
            result[cond] = float(probs[prob_idx])
            prob_idx += 1
    return result