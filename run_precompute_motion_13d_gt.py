"""Pre-compute 13D ego-compensated motion vectors per (video, GT obj_id, frame).

Mirrors run_precompute_motion_13d.py but reads bbox sequences from
Refer-KITTI_labels.json instead of YOLOv8-NS predict.txt. Used for FiLM
training where RMOT_Dataset keys by GT obj_id (Track_Dataset still keys
by NS tracker id at inference time, served from the NS-derived cache).

Output: /home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1/{video}.pt
        dict[obj_id: int][frame_id: int] = np.ndarray((13,), dtype=float32)

Usage:
    conda activate RMOT
    python run_precompute_motion_13d_gt.py
    python run_precompute_motion_13d_gt.py --seqs 0001 0002
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, "/home/seanachan/GMC-Link")
sys.path.insert(0, "/home/seanachan/iKUN")
from gmc_link.manager import GMCLinkManager
from utils import RESOLUTION

LABELS_JSON = "/home/seanachan/GMC-Link/Refer-KITTI_labels.json"
KITTI_IMG = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
OUT_ROOT = "/home/seanachan/GMC-Link/iKUN/motion_13d_cache_v1"

# iKUN VIDEOS["train"] (15 seqs); test {0005,0011,0013} stay NS-derived
TRAIN_SEQS = ["0001","0002","0003","0004","0006","0007","0008","0009","0010","0012",
              "0014","0015","0016","0018","0020"]


class _Track:
    __slots__ = ("id", "centroid", "bbox")
    def __init__(self, tid, centroid, bbox):
        self.id = tid
        self.centroid = centroid
        self.bbox = bbox


def _load_gt_tracks(labels, video):
    """Return (out, n_obj, n_rows) where out = dict[fid: int][oid: int] = (x,y,w,h) px."""
    out = defaultdict(dict)
    if video not in labels:
        return out, 0, 0
    H, W = RESOLUTION[video]
    n_rows = 0
    for oid_str, frames in labels[video].items():
        oid = int(oid_str)
        for fid_str, rec in frames.items():
            fid = int(fid_str)
            x_n, y_n, w_n, h_n = rec["bbox"]
            x, y, w, h = x_n * W, y_n * H, w_n * W, h_n * H
            out[fid][oid] = (x, y, w, h)
            n_rows += 1
    return out, len(labels[video]), n_rows


def _process_video(video, labels, manager):
    tracks_per_frame, n_obj, n_rows = _load_gt_tracks(labels, video)
    print(f"  [{video}] GT obj_ids={n_obj}, rows={n_rows}")
    if not tracks_per_frame:
        print(f"  [{video}] no GT tracks, skip")
        return {}

    cache = defaultdict(dict)  # obj_id -> {frame_id: np.ndarray(13,)}
    img_dir = Path(KITTI_IMG) / video
    frame_ids = sorted(tracks_per_frame.keys())
    dummy_lang = torch.zeros(1, 384)

    for fid in tqdm(frame_ids, desc=f"  {video}"):
        # GT json frame_ids are 0-indexed; KITTI image filenames are 0-indexed
        img_path = img_dir / f"{fid:06d}.png"
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
    p.add_argument("--seqs", nargs="+", default=TRAIN_SEQS)
    args = p.parse_args()

    with open(LABELS_JSON) as f:
        labels = json.load(f)

    os.makedirs(OUT_ROOT, exist_ok=True)
    for seq in args.seqs:
        print(f"=== {seq} ===")
        manager = GMCLinkManager()
        cache = _process_video(seq, labels, manager)
        n_obj = len(cache)
        n_vec = sum(len(v) for v in cache.values())
        if n_obj == 0:
            print(f"  [{seq}] empty cache, not writing .pt")
            continue
        out = os.path.join(OUT_ROOT, f"{seq}.pt")
        torch.save(cache, out)
        print(f"  wrote {out} ({n_obj} obj_ids, {n_vec} vecs)")


if __name__ == "__main__":
    main()
