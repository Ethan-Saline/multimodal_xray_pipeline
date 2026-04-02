import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
from health_multimodal.image.utils import ImageModelType, get_image_inference

from storage.cache import cache_embedding
from storage.db import (
    get_cursor,
    insert_embedding,
    insert_failed_encoding,
)
from storage.faiss_store import FAISSStore

warnings.filterwarnings("ignore", category=FutureWarning)

MANIFEST_PATH = Path("data/raw/manifest.csv")

def load_model():
    print("Loading BioViL encoder...")
    engine = get_image_inference(ImageModelType.BIOVIL)
    print("BioViL ready.")
    return engine

def get_layer4_module(engine):
    """
    Traverses the BioViL hierarchy based on your specific debug output.
    Path: engine.model -> .encoder (ImageEncoder) -> .encoder (ResNet) -> .layer4
    """
    try:
        return engine.model.encoder.encoder.layer4
    except AttributeError:
        # Fallback deep search
        for name, module in engine.model.named_modules():
            if name.endswith("encoder.layer4"):
                return module
        raise AttributeError("Could not find layer4 in the model hierarchy.")

def encode_dicom(engine, dicom_path: Path) -> tuple[np.ndarray, np.ndarray]:
    captured = {}

    def _hook(module, input, output):
        captured["feature_map"] = output.detach().cpu()

    target_layer = get_layer4_module(engine)
    handle = target_layer.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            embedding = engine.get_projected_global_embedding(dicom_path)
            if "feature_map" not in captured:
                raise RuntimeError("Forward hook failed to capture feature_map.")
    finally:
        handle.remove()

    emb_np = embedding.cpu().numpy().astype(np.float32)
    fmap = captured["feature_map"].numpy().astype(np.float32)
    if fmap.ndim == 4:
        fmap = fmap.squeeze(0)

    return emb_np, fmap

def encode_study(engine, dicom_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    embeddings, fmaps = [], []
    for p in dicom_paths:
        emb, fmap = encode_dicom(engine, p)
        embeddings.append(emb)
        fmaps.append(fmap)
    return np.mean(embeddings, axis=0).astype(np.float32), np.mean(fmaps, axis=0).astype(np.float32)

def run(split: str = None, limit: int = None):
    engine = load_model()
    store = FAISSStore()
    
    studies = {}
    with open(MANIFEST_PATH, "r") as f:
        for row in csv.DictReader(f):
            sid = row["study_id"]
            if split and row["split"] != split: continue
            if sid not in studies:
                studies[sid] = {"sid": sid, "paths": []}
            p = Path(row["dicom_path"])
            if p.exists(): studies[sid]["paths"].append(p)

    with get_cursor() as cur:
        cur.execute("SELECT study_id FROM embeddings")
        done = {r["study_id"] for r in cur.fetchall()}
    
    pending = [s for s in studies.values() if s["sid"] not in done]
    if limit: pending = pending[:limit]

    print(f"Studies to encode : {len(pending)}")
    success, failed = 0, 0

    for i, study in enumerate(pending):
        sid = study["sid"]
        paths = study["paths"]
        try:
            emb, fmap = encode_study(engine, paths)
            
            # FIX: Changed 'add_vector' to 'add' to match standard FAISSStore implementations
            # If this still fails, check storage/faiss_store.py for the correct method name
            faiss_id = store.add(emb) 
            
            insert_embedding(
                study_id=sid,
                dicom_uid=f"s{sid}", 
                faiss_id=faiss_id,
                model_name="biovil",
                embed_dim=emb.shape[0],
                feature_map=fmap.tobytes(), 
                feature_map_shape=list(fmap.shape)
            )
            cache_embedding(f"img_{sid}", emb)
            success += 1
        except Exception as e:
            print(f"  ERROR on study {sid}: {e}")
            insert_failed_encoding(sid, str(paths[0]) if paths else "N/A", str(e))
            failed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            print(f"  [{i+1}/{len(pending)}] success: {success} failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(split=args.split, limit=args.limit)