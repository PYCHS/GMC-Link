"""
Generate tracking visualization videos from GMC-Link prediction results.

Reads KITTI frames, draws GT (cyan) and prediction (green) bounding boxes,
and writes an MP4 video for each specified expression into video/.

Usage:
    python generate_video.py [--seq SEQ] [--expr EXPRESSION]
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np


# ── Paths ────────────────────────────────────────────────────────────

KITTI_IMG_ROOT = "refer-kitti/KITTI/training/image_02"
RESULT_ROOT_GMC = "/home/seanachan/TempRMOT/exps/default_rk/results_epoch0"
RESULT_ROOT_BASE = "/home/seanachan/TempRMOT/exps/default_rk/results_epoch50"
VIDEO_OUT = "video"

# Colors (BGR)
COLOR_GT = (255, 255, 0)       # Cyan – Ground Truth
COLOR_PRED = (0, 220, 0)       # Green – GMC-Link prediction
COLOR_BASE = (0, 120, 255)     # Orange – Baseline prediction
COLOR_TEXT_BG = (30, 30, 30)
COLOR_WHITE = (255, 255, 255)


# ── Helpers ──────────────────────────────────────────────────────────

def load_mot(filepath):
    """Load MOTChallenge-format file → {frame_id: [(track_id, x, y, w, h), ...]}"""
    data = defaultdict(list)
    if not os.path.isfile(filepath):
        return data
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data[fid].append((tid, x, y, w, h))
    return data


def draw_boxes(frame, boxes, color, label_prefix="", thickness=2):
    """Draw bounding boxes with track ID labels."""
    for tid, x, y, w, h in boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{label_prefix}#{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def draw_header(frame, text, frame_idx, total, n_gt, n_pred):
    """Draw a semi-transparent header bar with expression text and stats."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f'Query: "{text}"', (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    info = f"Frame {frame_idx+1}/{total}  |  GT: {n_gt}  Pred: {n_pred}"
    cv2.putText(frame, info, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_legend(frame):
    """Draw a small legend in the bottom-left corner."""
    h, w = frame.shape[:2]
    y0 = h - 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (260, h), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(frame, (10, y0 + 8), (30, y0 + 20), COLOR_GT, -1)
    cv2.putText(frame, "Ground Truth", (35, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (10, y0 + 28), (30, y0 + 40), COLOR_PRED, -1)
    cv2.putText(frame, "GMC-Link Prediction", (35, y0 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────

def generate_video(seq, expression, fps=10):
    """Generate a single comparison video for the given sequence and expression."""
    frame_dir = os.path.join(KITTI_IMG_ROOT, seq)
    gt_path = os.path.join(RESULT_ROOT_GMC, seq, expression, "gt.txt")
    pred_path = os.path.join(RESULT_ROOT_GMC, seq, expression, "predict.txt")

    if not os.path.isdir(frame_dir):
        print(f"[SKIP] Frame dir not found: {frame_dir}")
        return
    if not os.path.isfile(gt_path):
        print(f"[SKIP] GT not found: {gt_path}")
        return

    gt_data = load_mot(gt_path)
    pred_data = load_mot(pred_path)

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg")))
    if not frame_files:
        print(f"[SKIP] No frames in {frame_dir}")
        return

    # Read first frame for dimensions
    sample = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    h, w = sample.shape[:2]

    out_name = f"{seq}_{expression}.mp4"
    out_path = os.path.join(VIDEO_OUT, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    sentence = expression.replace("-", " ")

    for idx, fname in enumerate(frame_files):
        fpath = os.path.join(frame_dir, fname)
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        fid = idx + 1  # MOTChallenge is 1-indexed

        gt_boxes = gt_data.get(fid, [])
        pred_boxes = pred_data.get(fid, [])

        draw_boxes(frame, gt_boxes, COLOR_GT, label_prefix="GT", thickness=2)
        draw_boxes(frame, pred_boxes, COLOR_PRED, label_prefix="P", thickness=2)
        draw_header(frame, sentence, idx, len(frame_files), len(gt_boxes), len(pred_boxes))
        draw_legend(frame)

        writer.write(frame)

    writer.release()
    print(f"[DONE] {out_path}  ({len(frame_files)} frames, {len(frame_files)/fps:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate GMC-Link tracking visualization videos")
    parser.add_argument("--seq", type=str, default=None, help="Sequence ID (e.g. 0011)")
    parser.add_argument("--expr", type=str, default=None, help="Expression name (e.g. moving-vehicles)")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    args = parser.parse_args()

    os.makedirs(VIDEO_OUT, exist_ok=True)

    if args.seq and args.expr:
        generate_video(args.seq, args.expr, args.fps)
    else:
        # Default: generate videos for representative expressions across sequences
        demos = [
            ("0011", "moving-vehicles"),
            ("0011", "parking-vehicles"),
            ("0011", "vehicles-in-the-same-direction-of-ours"),
            ("0005", "moving-vehicles"),
            ("0005", "vehicles-in-the-same-direction-of-ours"),
            ("0013", "walking-women"),
        ]
        for seq, expr in demos:
            if os.path.isdir(os.path.join(RESULT_ROOT_GMC, seq, expr)):
                generate_video(seq, expr, args.fps)
            else:
                print(f"[SKIP] No results for {seq}/{expr}")


if __name__ == "__main__":
    main()
