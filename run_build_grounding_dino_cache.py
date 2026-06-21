"""Build Grounding-DINO-tiny detection cache for V1 test seqs (0005, 0011, 0013).

Output schema matches det_cache/DDETR-kitti/:
    {seq, class, img_h, img_w, score_thr, text_thr, schema,
     frames: {fid_1indexed: [[x1, y1, x2, y2, score], ...]}}

Per Path A spec (2026-05-16). Phase A1 detector cache build.

Usage:
    python run_build_grounding_dino_cache.py \
        --seqs 0005 0011 0013 \
        --out_root det_cache/grounding_dino_v1
"""
import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


KITTI_ROOT = "/home/seanachan/GMC-Link/refer-kitti/KITTI/training/image_02"
PROMPT = "a car. a person."
LABEL_TO_CLASS = {"a car": "car", "a person": "pedestrian"}
CLASSES = ["car", "pedestrian"]


def build_seq(model, proc, device, seq, out_root, box_thr, text_thr):
    img_dir = os.path.join(KITTI_ROOT, seq)
    frames = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    per_class = {c: {} for c in CLASSES}
    img_h = img_w = None

    for fname in tqdm(frames, desc=f"{seq}", leave=False):
        fid = int(fname[:-4]) + 1  # 1-indexed to match NS predict.txt
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        if img_h is None:
            img_w, img_h = img.size
        inputs = proc(images=img, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids,
            threshold=box_thr, text_threshold=text_thr,
            target_sizes=[img.size[::-1]],
        )[0]
        boxes = res["boxes"].cpu().tolist()
        scores = res["scores"].cpu().tolist()
        text_labels = res["text_labels"]
        for box, score, lab in zip(boxes, scores, text_labels):
            cls = LABEL_TO_CLASS.get(lab)
            if cls is None:
                continue
            per_class[cls].setdefault(str(fid), []).append(
                [float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)]
            )

    for cls in CLASSES:
        out_dir = os.path.join(out_root, seq, cls)
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "seq": seq, "class": cls,
            "img_h": img_h, "img_w": img_w,
            "score_thr": box_thr, "text_thr": text_thr,
            "schema": "frame_id (1-indexed) -> [[x1, y1, x2, y2, score], ...]",
            "frames": per_class[cls],
        }
        out_path = os.path.join(out_dir, "dets.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)
        n_boxes = sum(len(v) for v in per_class[cls].values())
        n_frames = len(per_class[cls])
        print(f"  {seq}/{cls}: {n_frames} frames, {n_boxes} boxes → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", nargs="+", default=["0005", "0011", "0013"])
    p.add_argument("--out_root", default="det_cache/grounding_dino_v1")
    p.add_argument("--ckpt", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--box_thr", type=float, default=0.25)
    p.add_argument("--text_thr", type=float, default=0.20)
    args = p.parse_args()

    device = "cuda"
    print(f"Loading {args.ckpt}...", flush=True)
    proc = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.ckpt).to(device).eval()
    print("Loaded.", flush=True)

    for seq in args.seqs:
        build_seq(model, proc, device, seq, args.out_root, args.box_thr, args.text_thr)


if __name__ == "__main__":
    main()
