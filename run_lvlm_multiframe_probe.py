"""C1 multi-frame probe: pass [crop@t-Δ, crop@t, crop@t+Δ] to Qwen2-VL.

Tests if temporal context lets LVLM judge motion. Δ=5 frames typical.
Tracks loaded by ID so same object across 3 frames.
"""
import argparse
import os
import re
import time
from collections import defaultdict

import torch
from PIL import Image


KITTI_ROOT = "/home/seanachan/GMC-Link/refer-kitti/KITTI/training/image_02"
NS_ROOT = "/home/seanachan/GMC-Link/NeuralSORT"


def parse_score(text):
    m = re.search(r"\b([0-9]|10)\b", text)
    if m is None:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 10 else None


def crop_box(img, box, pad_frac=0.20):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = pad_frac * w, pad_frac * h
    return img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                     min(img.width, int(x2 + px)), min(img.height, int(y2 + py))))


def load_tracks_by_id(seq, cls):
    """Return {track_id: {fid: xyxy}} from NS predict.txt."""
    path = os.path.join(NS_ROOT, seq, cls, "predict.txt")
    out = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            out[tid][fid] = (x, y, x + w, y + h)
    return out


def sample_triplets(tracks_by_id, delta, n_samples, seed=0):
    """Sample N (track_id, fid_center) where fid_center±delta both exist."""
    import random
    rng = random.Random(seed)
    candidates = []
    for tid, frames in tracks_by_id.items():
        fids = sorted(frames.keys())
        for f in fids:
            if (f - delta) in frames and (f + delta) in frames:
                candidates.append((tid, f))
    rng.shuffle(candidates)
    return candidates[:n_samples]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="0011")
    p.add_argument("--cls", default="car")
    p.add_argument("--expr", default="moving cars")
    p.add_argument("--n_calls", type=int, default=30)
    p.add_argument("--delta", type=int, default=5)
    p.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    args = p.parse_args()

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    print("Loading model...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb, device_map="auto"
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model_id)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    tracks_by_id = load_tracks_by_id(args.seq, args.cls)
    triplets = sample_triplets(tracks_by_id, args.delta, args.n_calls)
    print(f"Triplets: {len(triplets)} (n_tracks={len(tracks_by_id)})", flush=True)

    prompt = (
        f"These three images show the same object at consecutive moments "
        f"(Δ={args.delta} frames apart). Does this object's motion match "
        f"the description \"{args.expr}\"? "
        f"Score 0 (no match) to 10 (perfect match). Respond with a single integer."
    )

    img_dir = os.path.join(KITTI_ROOT, args.seq)
    scores = []
    t0 = time.time()
    for i, (tid, fc) in enumerate(triplets):
        crops = []
        for f in [fc - args.delta, fc, fc + args.delta]:
            img_path = os.path.join(img_dir, f"{f - 1:06d}.png")
            if not os.path.exists(img_path):
                break
            img = Image.open(img_path).convert("RGB")
            crops.append(crop_box(img, tracks_by_id[tid][f]))
        if len(crops) != 3:
            continue

        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": c} for c in crops],
            {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=crops, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        resp = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        s = parse_score(resp)
        if s is not None:
            scores.append(s)
        if i < 8:
            print(f"  [{i+1}] tid={tid} fc={fc} resp='{resp}' score={s}", flush=True)

    elapsed = time.time() - t0
    n = len(triplets)
    print()
    print(f"=== Multi-frame result ({args.expr}) ===")
    print(f"calls: {n}, parsed: {len(scores)} ({100*len(scores)/n:.1f}%)")
    print(f"latency: {elapsed/n:.2f} s/call")
    if scores:
        from collections import Counter
        h = Counter(scores)
        print(f"histogram: {dict(sorted(h.items()))}")
        print(f"mean={sum(scores)/len(scores):.2f}, n_unique={len(h)}")
    print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
