"""
Stage 3: Pre-compute Motion Embeddings for iKUN Feature-Level Injection
========================================================================
For each video in Refer-KITTI, runs GMC-Link's ego-motion compensation
on GT bounding boxes and extracts 256D motion embeddings (expression-independent).

Output: {video}_{obj_id} → {frame_id: 256D tensor}  saved as .pt files.

Usage:
    python gmc_link/precompute_motion_embeddings.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

import cv2

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.demo_inference import DummyTrack


RESOLUTION = {
    "0001": (375, 1242), "0002": (375, 1242), "0003": (375, 1242), "0004": (375, 1242),
    "0005": (375, 1242), "0006": (375, 1242), "0007": (375, 1242), "0008": (375, 1242),
    "0009": (375, 1242), "0010": (375, 1242), "0011": (375, 1242), "0012": (375, 1242),
    "0013": (375, 1242), "0014": (370, 1224), "0015": (370, 1224), "0016": (370, 1224),
    "0018": (374, 1238), "0020": (376, 1241),
}

TRAIN_VIDEOS = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]

TEST_VIDEOS = ["0011"]


def precompute_embeddings(
    videos: list,
    labels_path: str = "Refer-KITTI_labels.json",
    data_root: str = "refer-kitti",
    weights_path: str = "gmc_link_weights.pth",
    output_dir: str = "gmc_link/motion_embeddings",
) -> None:
    """
    Pre-compute 256D motion embeddings for all GT objects.

    For each video, processes ALL frames sequentially through GMC-Link's
    ego-motion compensation pipeline, then extracts the 256D embedding
    from the motion_projector (expression-independent).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    with open(labels_path, "r") as f:
        all_labels = json.load(f)

    # We only need the motion_projector, but we use GMCLinkManager for its
    # ego-motion pipeline. Use a dummy language embedding for process_frame.
    encoder = TextEncoder(device=device)
    dummy_lang = encoder.encode("dummy placeholder")

    for video in videos:
        if video not in all_labels:
            print(f"  {video}: no labels, skipping")
            continue

        H, W = RESOLUTION[video]
        video_labels = all_labels[video]

        # Build per-frame object lists from GT labels
        # obj_id → {frame_id: (x1_px, y1_px, w_px, h_px)}
        frame_objects = defaultdict(dict)  # frame_id → {obj_id: bbox}
        all_frame_ids = set()

        for obj_id_str, obj_data in video_labels.items():
            obj_id = int(obj_id_str)
            for frame_id_str, frame_data in obj_data.items():
                frame_id = int(frame_id_str)
                if "bbox" not in frame_data or not frame_data.get("category"):
                    continue
                x_n, y_n, w_n, h_n = frame_data["bbox"]
                x_px = x_n * W
                y_px = y_n * H
                w_px = w_n * W
                h_px = h_n * H
                frame_objects[frame_id][obj_id] = (x_px, y_px, w_px, h_px)
                all_frame_ids.add(frame_id)

        if not all_frame_ids:
            print(f"  {video}: no valid frames, skipping")
            continue

        # Initialize fresh GMC-Link manager per video
        linker = GMCLinkManager(weights_path=weights_path, device=device, lang_dim=384)

        # Image directory
        img_dir = os.path.join(data_root, "KITTI", "training", "image_02", video)
        img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        total_frames = len(img_files)

        # Storage: obj_id → {frame_id: 256D tensor}
        embeddings = defaultdict(dict)

        print(f"  {video}: {total_frames} frames, {len(video_labels)} objects")

        for frame_idx in tqdm(range(total_frames), desc=f"  {video}", leave=False):
            frame_path = os.path.join(img_dir, img_files[frame_idx])
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                continue

            objs = frame_objects.get(frame_idx, {})
            if not objs:
                # Still process frame for ego-motion tracking (empty tracks)
                linker.process_frame(frame_img, [], dummy_lang, detections=None)
                continue

            # Build tracks from GT bboxes
            active_tracks = []
            det_list = []
            for obj_id, (x, y, w, h) in objs.items():
                active_tracks.append(DummyTrack(obj_id, x, y, w, h))
                det_list.append([x, y, x + w, y + h])

            det_array = np.array(det_list) if det_list else None

            # Run GMC-Link — we need the velocities (8D vectors)
            _, velocities = linker.process_frame(
                frame_img, active_tracks, dummy_lang, detections=det_array
            )

            if not velocities:
                continue

            # Project 8D vectors through motion_projector → 256D
            motion_vectors = []
            obj_ids = []
            for obj_id, vel in velocities.items():
                motion_vectors.append(vel)
                obj_ids.append(obj_id)

            motion_tensor = torch.tensor(
                np.array(motion_vectors), dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                # Just the motion projector (no language needed)
                motion_emb = linker.aligner.motion_projector(motion_tensor)
                motion_emb = torch.nn.functional.normalize(motion_emb, p=2, dim=-1)

            for i, obj_id in enumerate(obj_ids):
                embeddings[obj_id][frame_idx] = motion_emb[i].cpu()

        # Save per-video
        output_path = os.path.join(output_dir, f"{video}.pt")
        torch.save(dict(embeddings), output_path)
        n_entries = sum(len(v) for v in embeddings.values())
        print(f"  {video}: saved {n_entries} embeddings → {output_path}")


if __name__ == "__main__":
    print("Pre-computing motion embeddings for TRAIN + TEST videos...")
    precompute_embeddings(TRAIN_VIDEOS + TEST_VIDEOS)
    print("Done!")
