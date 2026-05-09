"""Build per-track Z time-series cache via Depth Anything V2.

Sample metric Z at bbox center (5x5 patch median) for every track in every
frame of a sequence. Writes JSON: {track_id_str: {frame_id_str: z_meters}}.

Track sources:
  - ikun: NeuralSORT/{seq}/{car,pedestrian}/predict.txt (merged like run_build_gmc_cache)
  - flexhook_v1, flexhook_v2_raw: same NeuralSORT output for now (smoke smoke);
    arch arg only switches output filename to keep evals separable.

Usage:
    python run_build_depth_cache.py --arch ikun --seq 0011
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.demo_inference import load_neuralsort_tracks
from gmc_link.depth_cache import save_depth_cache
from gmc_link.depth_extractor import DepthExtractor

FRAME_DIR = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
TRACK_DIR = "NeuralSORT"


def merged_ns(seq: str) -> dict:
    car = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "car", "predict.txt"))
    ped = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "pedestrian", "predict.txt"))
    max_car = 0
    for _, dets in car.items():
        for oid, *_ in dets:
            max_car = max(max_car, oid)
    ns = defaultdict(list)
    for fid, dets in car.items():
        ns[fid].extend(dets)
    for fid, dets in ped.items():
        ns[fid].extend([(oid + max_car, x, y, w, h) for oid, x, y, w, h in dets])
    return ns


def patch_z(depth: np.ndarray, cx: int, cy: int, half: int = 2) -> float:
    H, W = depth.shape
    cx = int(np.clip(cx, half, W - 1 - half))
    cy = int(np.clip(cy, half, H - 1 - half))
    patch = depth[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return float(np.median(patch))


def build(arch: str, seq: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"[depth] cache exists → {out_path}, skip")
        return

    extractor = DepthExtractor(device="cuda")
    ns = merged_ns(seq)
    seq_frame_dir = os.path.join(FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))
    total = len(frame_files)
    print(f"[depth] arch={arch} seq={seq} frames={total} tracks_per_frame_avg≈{sum(len(v) for v in ns.values()) / max(1, total):.1f}")

    table: dict[str, dict[str, float]] = defaultdict(dict)
    for f0 in tqdm(range(total), desc=f"depth-{arch}-{seq}"):
        f1 = f0 + 1
        dets = ns.get(f1, [])
        if not dets:
            continue
        bgr = cv2.imread(os.path.join(seq_frame_dir, frame_files[f0]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = extractor.extract(rgb)
        for oid, x, y, w, h in dets:
            cx = x + w / 2.0
            cy = y + h / 2.0
            z = patch_z(depth, int(round(cx)), int(round(cy)))
            table[str(oid)][str(f1)] = z

    save_depth_cache(table, out_path)
    n_tracks = len(table)
    n_pts = sum(len(v) for v in table.values())
    print(f"[depth] wrote {out_path}  tracks={n_tracks}  samples={n_pts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ikun", "flexhook_v1", "flexhook_v2_raw"])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"gmc_link/z_track_{args.arch}_{args.seq}_cache.json"
    build(args.arch, args.seq, out)
