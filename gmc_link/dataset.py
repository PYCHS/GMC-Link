"""
Dataset generation for GMC-Link: spatial-temporal feature extraction from Refer-KITTI targets.
(Modified for motion representation ablation: baseline / camera / relative)
"""

import json
import os
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gmc_link.utils import VELOCITY_SCALE, warp_points
from gmc_link.core import ORBHomographyEngine

HOMOGRAPHY_CACHE = {}

# Multi-scale frame gaps for temporal velocity features
FRAME_GAPS = [2, 5, 10]  # short, mid, long


class MotionLanguageDataset(Dataset):
    def __init__(self, motion_data, language_data, labels):
        assert len(motion_data) == len(language_data) == len(labels)
        self.motion_data = motion_data
        self.language_data = language_data
        self.labels = labels

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        motion = torch.tensor(self.motion_data[idx], dtype=torch.float32)
        lang = torch.tensor(self.language_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return motion, lang, label


def collate_fn(batch):
    motion_batch = torch.stack([item[0] for item in batch], dim=0)
    language_batch = torch.stack([item[1] for item in batch], dim=0)
    label_batch = torch.stack([item[2] for item in batch], dim=0)
    return motion_batch, language_batch, label_batch


# ─────────────────────────────────────────────


def load_refer_kitti_expressions(expression_dir):
    expressions = []
    for json_file in sorted(os.listdir(expression_dir)):
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(expression_dir, json_file), "r", encoding="utf-8") as f:
            expr = json.load(f)
        expressions.append(expr)
    return expressions


def load_labels_with_ids(labels_dir):
    labels = {}
    for txt_file in sorted(os.listdir(labels_dir)):
        if not txt_file.endswith(".txt"):
            continue
        frame_idx = int(os.path.splitext(txt_file)[0])
        frame_labels = []
        with open(os.path.join(labels_dir, txt_file), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                frame_labels.append(
                    {
                        "track_id": int(parts[1]),
                        "x1_n": float(parts[2]),
                        "y1_n": float(parts[3]),
                        "w_n": float(parts[4]),
                        "h_n": float(parts[5]),
                    }
                )
        labels[frame_idx] = frame_labels
    return labels


# ─────────────────────────────────────────────


def _collect_expressions(data_root, sequences, text_encoder):
    all_expressions = []
    sentence_embeddings = {}

    for seq in sequences:
        expression_dir = os.path.join(data_root, "expression", seq)
        if not os.path.exists(expression_dir):
            continue

        exprs = load_refer_kitti_expressions(expression_dir)
        for expr in exprs:
            sentence = expr["sentence"]

            if sentence not in sentence_embeddings:
                emb = text_encoder.encode(sentence).squeeze(0).cpu().numpy()
                sentence_embeddings[sentence] = emb

            all_expressions.append(
                {
                    "sentence": sentence,
                    "embedding": sentence_embeddings[sentence],
                    "label": expr["label"],
                    "seq": seq,
                }
            )

    return all_expressions, sentence_embeddings, list(sentence_embeddings.keys())


def _extract_target_centroids(data_root, seq, label_map, frame_shape):
    labels_dir = os.path.join(data_root, "KITTI", "labels_with_ids", "image_02", seq)
    if not os.path.exists(labels_dir):
        return {}

    labels_by_frame = load_labels_with_ids(labels_dir)
    h, w = frame_shape

    track_centroids = {}
    for fid in label_map:
        fid = int(fid)
        if fid not in labels_by_frame:
            continue

        for det in labels_by_frame[fid]:
            tid = det["track_id"]

            x1 = det["x1_n"] * w
            y1 = det["y1_n"] * h
            bw = det["w_n"] * w
            bh = det["h_n"] * h

            cx = x1 + bw / 2
            cy = y1 + bh / 2

            if tid not in track_centroids:
                track_centroids[tid] = {}

            track_centroids[tid][fid] = (cx, cy, bw, bh)

    return track_centroids


# ─────────────────────────────────────────────
# ⭐ 核心：motion representation
# ─────────────────────────────────────────────


def _compute_motion(
    cx1, cy1, bw1, bh1,
    cx2, cy2, bw2, bh2,
    homography,
    frame_shape,
    motion_mode
):
    h, w = frame_shape

    dx = (cx2 - cx1) / w * VELOCITY_SCALE
    dy = (cy2 - cy1) / h * VELOCITY_SCALE
    dw = (bw2 - bw1) / w * VELOCITY_SCALE
    dh = (bh2 - bh1) / h * VELOCITY_SCALE

    cx_n, cy_n = cx1 / w, cy1 / h
    bw_n, bh_n = bw1 / w, bh1 / h

    base = [dx, dy, dw, dh, cx_n, cy_n, bw_n, bh_n]

    tx, ty = 0.0, 0.0
    if homography is not None:
        tx = homography[0, 2] / w
        ty = homography[1, 2] / h

    dx_rel = dx - tx
    dy_rel = dy - ty

    if motion_mode == "baseline":
        return np.array(base, dtype=np.float32)

    elif motion_mode == "camera":
        return np.array(base + [tx, ty], dtype=np.float32)

    elif motion_mode == "relative":
        return np.array(base + [tx, ty, dx_rel, dy_rel], dtype=np.float32)

    else:
        raise ValueError("Unknown motion_mode")


# ─────────────────────────────────────────────


def _generate_positive_pairs(
    track_centroids,
    embedding,
    expression_id,
    frame_gap,
    frame_shape,
    seq=None,
    frame_dir=None,
    orb_engine=None,
    motion_mode="baseline"
):
    motion_data, language_data, labels = [], [], []

    primary_gap_idx = 1  # mid-scale (gap=5) is the primary for dw/dh/snr

    for tid, centroids in track_centroids.items():
        frames = sorted(centroids.keys())

        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]

            cx1, cy1, bw1, bh1 = centroids[f1]
            cx2, cy2, bw2, bh2 = centroids[f2]

            homography = None
            if frame_dir and orb_engine:
                key = (seq, f1, f2)
                if key not in HOMOGRAPHY_CACHE:
                    img1 = cv2.imread(os.path.join(frame_dir, f"{f1:06d}.png"))
                    img2 = cv2.imread(os.path.join(frame_dir, f"{f2:06d}.png"))
                    if img1 is not None and img2 is not None:
                        HOMOGRAPHY_CACHE[key] = orb_engine.estimate_homography(img1, img2)
                homography = HOMOGRAPHY_CACHE.get(key)

            motion_vec = _compute_motion(
                cx1, cy1, bw1, bh1,
                cx2, cy2, bw2, bh2,
                homography,
                frame_shape,
                motion_mode
            )

            motion_data.append(motion_vec)
            language_data.append(embedding.copy())
            labels.append(expression_id)

    return motion_data, language_data, labels


# ─────────────────────────────────────────────


def build_training_data(
    data_root,
    sequences,
    text_encoder,
    frame_gap=5,
    frame_shape=(375, 1242),
    motion_mode="baseline"
):
    all_expressions, _, sentences = _collect_expressions(
        data_root, sequences, text_encoder
    )

    sentence_to_id = {s: i for i, s in enumerate(sentences)}

    orb_engine = ORBHomographyEngine(max_features=1500)

    motion_data, language_data, labels = [], [], []

    # Filter to motion-relevant expressions only — appearance-only sentences
    # like "red cars" add noise since motion vectors can't encode color/shape
    motion_expressions = [e for e in all_expressions if is_motion_expression(e["sentence"])]
    print(f"  Motion-filtered: {len(motion_expressions)}/{len(all_expressions)} expressions")

    for expr in motion_expressions:
        seq = expr["seq"]
        label_map = expr["label"]
        embedding = expr["embedding"]
        expression_id = sentence_to_id[expr["sentence"]]

        frame_dir = os.path.join(data_root, "KITTI", "training", "image_02", seq)

        track_centroids = _extract_target_centroids(
            data_root, seq, label_map, frame_shape
        )

        m, l, lbl = _generate_positive_pairs(
            track_centroids,
            embedding,
            expression_id,
            frame_gap,
            frame_shape,
            seq,
            frame_dir,
            orb_engine,
            motion_mode
        )

        motion_data.extend(m)
        language_data.extend(l)
        labels.extend(lbl)

    print(f"Total samples: {len(labels)}")
    return motion_data, language_data, labels