-- migration_002_feature_map.sql
-- Adds feature_map and feature_map_shape to the embeddings table.
-- Run before re-running encode.py.
--
-- feature_map       : raw bytes of a (2048, H, W) float32 numpy array
--                     serialised with feature_map.tobytes()
-- feature_map_shape : e.g. {2048,7,7} — needed to reconstruct the array
--
-- Reconstruction in validate.py:
--     raw   = row["feature_map"]
--     shape = tuple(row["feature_map_shape"])
--     fmap  = np.frombuffer(bytes(raw), dtype=np.float32).reshape(shape)
--
-- Storage estimate:
--     2048 * 7 * 7 * 4 bytes = ~401 KB per study
--     At 5,000 studies       = ~1.9 GB total
--
-- Existing rows will have NULL in both columns.
-- Re-run encode.py to backfill, or accept NULLs for studies encoded
-- before this migration (heatmaps will not be available for those).

ALTER TABLE embeddings
    ADD COLUMN IF NOT EXISTS feature_map       BYTEA,
    ADD COLUMN IF NOT EXISTS feature_map_shape INTEGER[];
