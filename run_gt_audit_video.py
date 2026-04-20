#!/usr/bin/env python
"""Render GT-annotated audit videos for Refer-KITTI V1 motion expressions."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gmc_link.dataset import is_motion_expression

BASE = Path("/home/seanachan/GMC-Link")
DATA_ROOT = Path("/home/seanachan/data/Dataset/refer-kitti")
IMG_ROOT = DATA_ROOT / "KITTI" / "training" / "image_02"
EXPR_ROOT = DATA_ROOT / "expression"
GT_TEMPLATE = BASE / "Refer-KITTI" / "gt_template"
NEURALSORT_ROOT = BASE / "NeuralSORT"

OUT_DIR = BASE / "diagnostics" / "results" / "label_audit"

SEQS = ["0005", "0011"]
FPS = 5
COLOR_GT = (0, 220, 0)
COLOR_CONTEXT = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (30, 30, 30)


def load_expression(path: Path) -> dict:
    """Parse expression JSON into sentence and per-frame GT track IDs."""
    with open(path) as f:
        d = json.load(f)
    gt_by_frame = {int(fid): [int(t) for t in tids] for fid, tids in d["label"].items()}
    return {"sentence": d["sentence"], "gt_by_frame": gt_by_frame}


def load_gt_template_mot(seq: str, slug: str) -> dict:
    path = GT_TEMPLATE / seq / slug / "gt.txt"
    out: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(p) for p in parts[2:6])
            out[fid].append((tid, x, y, w, h))
    return out


def load_neuralsort_context(seq: str) -> dict:
    out: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    id_offset = 0
    for sub in ("car", "pedestrian"):
        path = NEURALSORT_ROOT / seq / sub / "predict.txt"
        if not path.is_file():
            continue
        try:
            arr = np.loadtxt(path, delimiter=",")
        except Exception:
            continue
        if arr.ndim != 2 or len(arr) == 0:
            continue
        max_id_here = int(arr[:, 1].max())
        for row in arr:
            fid = int(row[0])
            tid = int(row[1]) + id_offset
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            out[fid].append((tid, x, y, w, h))
        id_offset += max_id_here + 1
    return out


if __name__ == "__main__":
    print("Loaders only — run main() added in Task 4.")
