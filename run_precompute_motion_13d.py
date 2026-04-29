"""Pre-compute 13D ego-compensated motion vectors per (video, obj_id, frame).

Reads YOLOv8-NS tracks from /home/seanachan/GMC-Link/NeuralSORT/{seq}/{car,pedestrian}/predict.txt.
Runs GMCLinkManager.process_frame frame-by-frame to populate centroid history + cumulative
homographies, then extracts the 13D vector from velocities_dict.

Output: /home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1/{video}.pt
        dict[obj_id: int][frame_id: int] = np.ndarray((13,), dtype=float32)

Usage:
    conda activate RMOT
    python run_precompute_motion_13d.py
    python run_precompute_motion_13d.py --seqs 0005 0011 0013
"""
import argparse, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, "/home/seanachan/GMC-Link")
from gmc_link.manager import GMCLinkManager

NS_ROOT = "/home/seanachan/GMC-Link/NeuralSORT"
KITTI_IMG = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
OUT_ROOT = "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1"

TRAIN_SEQS = ["0001","0002","0003","0004","0006","0007","0008","0009","0010","0012",
              "0014","0015","0016","0017","0018","0019","0020"]
TEST_SEQS = ["0005", "0011", "0013"]


class _Track:
    __slots__ = ("id", "centroid", "bbox")
    def __init__(self, tid, centroid, bbox):
        self.id = tid
        self.centroid = centroid
        self.bbox = bbox


def _load_ns_tracks(video):
    """Return dict[frame_id: int][obj_id: int] = (x,y,w,h) merged car+pedestrian."""
    out = defaultdict(dict)
    car = os.path.join(NS_ROOT, video, "car", "predict.txt")
    ped = os.path.join(NS_ROOT, video, "pedestrian", "predict.txt")
    max_car = 0
    if os.path.exists(car) and os.path.getsize(car) > 0:
        rows = np.loadtxt(car, delimiter=",", ndmin=2)
        for r in rows:
            fid, oid, x, y, w, h = int(r[0]), int(r[1]), r[2], r[3], r[4], r[5]
            out[fid][oid] = (x, y, w, h)
            max_car = max(max_car, oid)
    if os.path.exists(ped) and os.path.getsize(ped) > 0:
        rows = np.loadtxt(ped, delimiter=",", ndmin=2)
        for r in rows:
            fid, oid, x, y, w, h = int(r[0]), int(r[1]) + max_car, r[2], r[3], r[4], r[5]
            out[fid][oid] = (x, y, w, h)
    return out


def _process_video(video, manager):
    tracks_per_frame = _load_ns_tracks(video)
    if not tracks_per_frame:
        print(f"  [{video}] no tracks, skip")
        return {}

    cache = defaultdict(dict)  # obj_id -> {frame_id: np.ndarray(13,)}
    img_dir = Path(KITTI_IMG) / video
    frame_ids = sorted(tracks_per_frame.keys())
    dummy_lang = torch.zeros(1, 384)

    for fid in tqdm(frame_ids, desc=f"  {video}"):
        # MOT 1-indexed; KITTI image filenames 0-indexed
        img_path = img_dir / f"{fid - 1:06d}.png"
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        per_obj = tracks_per_frame[fid]
        active = []
        det_arr = []
        for oid, (x, y, w, h) in per_obj.items():
            cx, cy = x + w / 2.0, y + h / 2.0
            active.append(_Track(oid, np.array([cx, cy], dtype=np.float64),
                                 (x, y, x + w, y + h)))
            det_arr.append([x, y, x + w, y + h])
        det_arr = np.array(det_arr, dtype=np.float32) if det_arr else None
        _, velocities_dict, _ = manager.process_frame(
            frame, active, dummy_lang, detections=det_arr, update_state=True
        )
        for oid, vec in velocities_dict.items():
            cache[oid][fid] = vec.astype(np.float32)
    return dict(cache)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", nargs="+", default=TRAIN_SEQS + TEST_SEQS)
    args = p.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    for seq in args.seqs:
        print(f"=== {seq} ===")
        manager = GMCLinkManager()
        cache = _process_video(seq, manager)
        n_obj = len(cache)
        n_vec = sum(len(v) for v in cache.values())
        if n_obj == 0:
            print(f"  [{seq}] empty cache, not writing .pt (NeuralSORT predict.txt missing or empty)")
            continue
        out = os.path.join(OUT_ROOT, f"{seq}.pt")
        torch.save(cache, out)
        print(f"  wrote {out} ({n_obj} obj_ids, {n_vec} vecs)")


if __name__ == "__main__":
    main()
