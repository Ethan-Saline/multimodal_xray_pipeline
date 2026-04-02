import os
import json
import pickle
import numpy as np
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB       = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

TTL_EMBEDDING = int(os.getenv("CACHE_TTL_EMBEDDING", str(60 * 60 * 24 * 7)))  # 7 days
TTL_GENERAL   = int(os.getenv("CACHE_TTL_GENERAL",   str(60 * 60 * 24)))       # 1 day


def _client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=False,   # raw bytes needed for numpy pickle
    )


# ------------------------------------------------------------------ #
# Embedding cache                                                      #
# ------------------------------------------------------------------ #

def cache_embedding(dicom_uid: str, embedding: np.ndarray, ttl: int = TTL_EMBEDDING):
    _client().setex(f"emb:{dicom_uid}", ttl, pickle.dumps(embedding))


def get_cached_embedding(dicom_uid: str) -> np.ndarray | None:
    data = _client().get(f"emb:{dicom_uid}")
    return pickle.loads(data) if data else None


def invalidate_embedding(dicom_uid: str):
    """Call this after marking a DICOM as deleted."""
    _client().delete(f"emb:{dicom_uid}")


# ------------------------------------------------------------------ #
# General JSON cache                                                   #
# ------------------------------------------------------------------ #

def cache_set(key: str, value: object, ttl: int = TTL_GENERAL):
    """
    Store any JSON-serialisable value.
    Suggested key patterns:
        report:{study_id}
        diagnosis:{study_id}
        retrieval:{study_id}
    """
    _client().setex(key, ttl, json.dumps(value).encode())


def cache_get(key: str) -> object | None:
    data = _client().get(key)
    return json.loads(data.decode()) if data else None


def cache_delete(key: str):
    _client().delete(key)


def cache_flush_prefix(prefix: str):
    """Delete all keys with a given prefix. Use carefully."""
    c    = _client()
    keys = c.keys(f"{prefix}*")
    if keys:
        c.delete(*keys)


# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #

def ping() -> bool:
    try:
        return _client().ping()
    except redis.ConnectionError:
        return False


# ------------------------------------------------------------------ #
# Smoke test — python cache.py                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Running Redis smoke test...")

    if not ping():
        print("  ERROR: Redis not reachable.")
        print("  Start it with: docker run -d -p 6379:6379 redis:7-alpine")
        exit(1)

    print("  Redis reachable ✓")

    uid = "test-dicom-001"
    vec = np.random.rand(1024).astype(np.float32)
    cache_embedding(uid, vec, ttl=60)
    out = get_cached_embedding(uid)
    assert out is not None and np.allclose(vec, out)
    print("  Embedding round-trip ✓")

    cache_set("report:99999", {"findings": "clear", "impression": "normal"}, ttl=60)
    val = cache_get("report:99999")
    assert val["impression"] == "normal"
    print("  JSON round-trip ✓")

    invalidate_embedding(uid)
    cache_delete("report:99999")
    print("Redis smoke test passed.")