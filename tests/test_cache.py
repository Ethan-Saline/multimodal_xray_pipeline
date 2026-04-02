import sys
import numpy as np
from storage.cache import (
    ping, cache_embedding, get_cached_embedding, invalidate_embedding,
    cache_set, cache_get, cache_delete,
)


def test_redis_reachable():
    assert ping(), "Redis not reachable — is the Docker container running?"
    print("  Redis reachable ✓")


def test_embedding_roundtrip():
    uid = "test-smoke-001"
    vec = np.random.rand(1024).astype(np.float32)

    cache_embedding(uid, vec, ttl=60)
    out = get_cached_embedding(uid)

    assert out is not None, "Embedding not found in cache"
    assert np.allclose(vec, out), "Embedding values changed after roundtrip"
    print("  Embedding cache roundtrip ✓")

    invalidate_embedding(uid)
    assert get_cached_embedding(uid) is None, "Embedding should be gone after invalidation"
    print("  Embedding invalidation ✓")


def test_json_roundtrip():
    key = "test:report:99999"
    val = {"findings": "clear lungs", "impression": "no acute process", "score": 0.91}

    cache_set(key, val, ttl=60)
    out = cache_get(key)

    assert out is not None, "Value not found in cache"
    assert out["score"] == 0.91
    assert out["impression"] == "no acute process"
    print("  JSON cache roundtrip ✓")

    cache_delete(key)
    assert cache_get(key) is None, "Key should be gone after delete"
    print("  JSON cache delete ✓")


def test_cache_miss_returns_none():
    result = get_cached_embedding("this-uid-does-not-exist")
    assert result is None
    print("  Cache miss returns None ✓")


def test_embedding_dim_preserved():
    """Verify 1024-dim vectors come back at the right shape."""
    uid = "test-dim-check"
    vec = np.random.rand(1024).astype(np.float32)
    cache_embedding(uid, vec, ttl=60)
    out = get_cached_embedding(uid)
    assert out.shape == (1024,), f"Expected shape (1024,), got {out.shape}"
    print(f"  Embedding shape preserved: {out.shape} ✓")
    invalidate_embedding(uid)


if __name__ == "__main__":
    tests = [
        test_redis_reachable,
        test_embedding_roundtrip,
        test_json_roundtrip,
        test_cache_miss_returns_none,
        test_embedding_dim_preserved,
    ]

    print("Running cache tests...\n")
    failed = 0
    for t in tests:
        try:
            print(f"[{t.__name__}]")
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) failed.'}")
    sys.exit(0 if not failed else 1)