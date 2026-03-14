"""
Stage 3 Evaluation: Generate iKUN+Motion results.json for demo_inference comparison.

Runs the fine-tuned iKUN model (with motion embeddings) on seq 0011,
producing a results.json file compatible with demo_inference.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/home/seanachan/iKUN")

import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from opts import opt as default_opt
from model import get_model
from utils import load_from_ckpt, tokenize, EXPRESSIONS, expression_conversion

from gmc_link.demo_inference import load_neuralsort_tracks, DummyTrack


def generate_results(
    ckpt_path: str,
    motion_emb_path: str,
    sequence: str = "0011",
    track_dir: str = "NeuralSORT",
    data_root: str = "refer-kitti",
    output_path: str = "iKUN/results_motion.json",
):
    """Generate iKUN+motion results.json for a sequence."""
    # Override opts
    opt_args = [
        "--save_root", "/home/seanachan/GMC-Link",
        "--motion_emb_dir", os.path.dirname(motion_emb_path),
        "--gpus", "0",
    ]
    from opts import opts
    opt = opts().parse(opt_args)

    print("Loading model...")
    model = get_model(opt, "Model")
    model, _ = load_from_ckpt(model, ckpt_path)
    model.eval()

    # Load motion embeddings for this sequence
    motion_embs = torch.load(motion_emb_path, map_location="cpu")
    print(f"Loaded motion embeddings: {sum(len(v) for v in motion_embs.values())} entries")

    # Load tracks
    track_path = os.path.join(track_dir, sequence, "car", "predict.txt")
    ns_tracks = load_neuralsort_tracks(track_path)

    # Image dir
    img_dir = os.path.join(data_root, "KITTI", "training", "image_02", sequence)
    from torchvision import transforms as T
    from PIL import Image

    # Transforms matching iKUN
    norm_mean = [0.48145466, 0.4578275, 0.40821073]
    norm_std = [0.26862954, 0.26130258, 0.27577711]

    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = max(w, h)
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            from torchvision.transforms import functional as F
            return F.pad(image, (hp, vp, hp, vp), 0, "constant")

    local_transform = T.Compose([
        SquarePad(), T.Resize((224, 224)), T.ToTensor(), T.Normalize(norm_mean, norm_std)])
    global_transform = T.Compose([
        SquarePad(), T.Resize((672, 672)), T.ToTensor(), T.Normalize(norm_mean, norm_std)])

    # Expressions
    expressions = EXPRESSIONS.get(sequence, EXPRESSIONS.get("test", []))
    dropped = set(EXPRESSIONS.get("dropped", []))
    expressions = [e for e in expressions if e not in dropped]
    print(f"Evaluating {len(expressions)} expressions")

    # Results: video → obj_id → frame_id → expression → [score]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    frame_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))

    with torch.no_grad():
        for frame_1idx in tqdm(sorted(ns_tracks.keys()), desc="Frames"):
            frame_0idx = frame_1idx - 1
            if frame_0idx >= len(frame_files):
                continue

            frame_path = os.path.join(img_dir, frame_files[frame_0idx])
            frame_img = Image.open(frame_path)

            detections = ns_tracks[frame_1idx]

            for obj_id, x, y, w, h in detections:
                # Crop for local image
                x1, y1, x2, y2 = x, y, x + w, y + h
                crop = frame_img.crop([x1, y1, x2, y2])
                local_img = local_transform(crop).unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
                global_img = global_transform(frame_img).unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]

                # Get motion embedding
                obj_embs = motion_embs.get(obj_id, {})
                if frame_0idx in obj_embs:
                    motion_emb = obj_embs[frame_0idx].unsqueeze(0)  # [1, 256]
                else:
                    motion_emb = torch.zeros(1, 256)

                # Batch all expressions for this object
                for expr in expressions:
                    expr_converted = expression_conversion(expr)
                    tokens = tokenize([expr_converted]).cuda()

                    inputs = dict(
                        local_img=local_img.cuda(),
                        global_img=global_img.cuda(),
                        exp=tokens,
                        motion_emb=motion_emb.cuda(),
                    )
                    score = model(inputs)["logits"].cpu().item()
                    results[sequence][str(obj_id)][str(frame_1idx)][expr].append(score)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    generate_results(
        ckpt_path="/home/seanachan/GMC-Link/iKUN_motion/epoch119.pth",
        motion_emb_path="/home/seanachan/GMC-Link/gmc_link/motion_embeddings/0011.pt",
    )
