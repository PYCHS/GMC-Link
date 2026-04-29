import os
import random

import torch
import numpy as np

CACHE_DIR = "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1"

def test_cache_schema_v1_test_seqs():
    rng = random.Random(0)
    for video in ("0005", "0011", "0013"):
        path = os.path.join(CACHE_DIR, f"{video}.pt")
        assert os.path.exists(path), f"missing {path}"
        cache = torch.load(path, map_location="cpu", weights_only=False)
        assert isinstance(cache, dict), "top-level must be dict[obj_id]"
        assert len(cache) > 0, f"{video} empty"
        first_obj_id = next(iter(cache.keys()))
        first_obj = cache[first_obj_id]
        assert isinstance(first_obj, dict), "second-level must be dict[frame_id]"
        first_frame = next(iter(first_obj.keys()))
        vec = first_obj[first_frame]
        assert isinstance(vec, np.ndarray), "leaf must be np.ndarray"
        assert vec.shape == (13,), f"expected (13,), got {vec.shape}"
        assert vec.dtype == np.float32, f"expected float32, got {vec.dtype}"
        assert np.isfinite(vec).all(), "non-finite values in vec"

        # Strengthened sampling: up to 5 random obj_ids x up to 3 random frames each.
        obj_ids = list(cache.keys())
        sampled_obj_ids = (
            rng.sample(obj_ids, 5) if len(obj_ids) >= 5 else list(obj_ids)
        )
        for oid in sampled_obj_ids:
            frames = list(cache[oid].keys())
            assert len(frames) > 0, f"{video} obj_id={oid} has no frames"
            sampled_frames = (
                rng.sample(frames, 3) if len(frames) >= 3 else list(frames)
            )
            for fid in sampled_frames:
                v = cache[oid][fid]
                assert isinstance(v, np.ndarray), (
                    f"{video} obj_id={oid} frame={fid} not ndarray"
                )
                assert v.shape == (13,), (
                    f"{video} obj_id={oid} frame={fid} shape {v.shape}"
                )
                assert v.dtype == np.float32, (
                    f"{video} obj_id={oid} frame={fid} dtype {v.dtype}"
                )
                assert np.isfinite(v).all(), (
                    f"{video} obj_id={oid} frame={fid} non-finite"
                )
