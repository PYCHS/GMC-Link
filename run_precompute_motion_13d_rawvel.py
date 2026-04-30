"""Ablation: 13D vector with RAW velocity (no ego-compensation) for FiLM.

Monkey-patches `GMCLinkManager.ego_engine.estimate_homography` to return identity,
so residual_velocity collapses to raw velocity. Runs both pipelines so the rawvel
cache covers both train (GT-keyed, 15 seqs) and test (NS-keyed, 3 seqs):
  - run_precompute_motion_13d_gt._process_video  → train seqs
  - run_precompute_motion_13d._process_video     → test seqs

Output: /home/seanachan/GMC-Link/iKUN/motion_13d_cache_rawvel_v1/{video}.pt
        same shape as motion_13d_cache_v1, channels 0-5 are RAW velocity.

Usage:
    conda activate RMOT
    python run_precompute_motion_13d_rawvel.py
"""
import argparse, json, os, sys, types
sys.path.insert(0, "/home/seanachan/GMC-Link")
import numpy as np
import torch

import run_precompute_motion_13d as ns_base
import run_precompute_motion_13d_gt as gt_base
from gmc_link.manager import GMCLinkManager

OUT_ROOT = "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_rawvel_v1"

# iKUN VIDEOS["train"] (15 seqs) and VIDEOS["test"] / val (3 seqs)
TRAIN_SEQS = ["0001","0002","0003","0004","0006","0007","0008","0009","0010","0012",
              "0014","0015","0016","0018","0020"]
TEST_SEQS = ["0005", "0011", "0013"]


def _identity_estimate(self, prev_frame, curr_frame, prev_dets):
    return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)


def _patched_manager():
    m = GMCLinkManager()
    m.ego_engine.estimate_homography = types.MethodType(_identity_estimate, m.ego_engine)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--test_seqs", nargs="+", default=TEST_SEQS)
    args = p.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Train (GT-keyed)
    with open(gt_base.LABELS_JSON) as f:
        labels = json.load(f)
    for seq in args.train_seqs:
        print(f"=== TRAIN {seq} (GT, rawvel) ===")
        manager = _patched_manager()
        cache = gt_base._process_video(seq, labels, manager)
        if not cache:
            print(f"  [{seq}] empty cache, skip"); continue
        out = os.path.join(OUT_ROOT, f"{seq}.pt")
        torch.save(cache, out)
        n_obj = len(cache); n_vec = sum(len(v) for v in cache.values())
        print(f"  wrote {out} ({n_obj} obj_ids, {n_vec} vecs)")

    # Test (NS-keyed)
    for seq in args.test_seqs:
        print(f"=== TEST {seq} (NS, rawvel) ===")
        manager = _patched_manager()
        cache = ns_base._process_video(seq, manager)
        if not cache:
            print(f"  [{seq}] empty cache, skip"); continue
        out = os.path.join(OUT_ROOT, f"{seq}.pt")
        torch.save(cache, out)
        n_obj = len(cache); n_vec = sum(len(v) for v in cache.values())
        print(f"  wrote {out} ({n_obj} obj_ids, {n_vec} vecs)")


if __name__ == "__main__":
    main()
