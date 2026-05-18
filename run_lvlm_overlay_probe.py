"""C1 whole-frame + bbox-overlay probe: LVLM sees scene + highlighted target.

Tests if scene context lets LVLM judge motion. Draws red bbox on full frame
at fid-Δ, fid, fid+Δ. Object motion relative to scene becomes visible.
"""
import argparse
import os
import re
import time
from collections import defaultdict

import torch
from PIL import Image, ImageDraw


KITTI_ROOT = "/home/seanachan/GMC-Link/refer-kitti/KITTI/training/image_02"
NS_ROOT = "/home/seanachan/GMC-Link/NeuralSORT"


def parse_score(text):
    m = re.search(r"\b([0-9]|10)\b", text)
    if m is None:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 10 else None


def draw_bbox(img, box, color="red", width=4):
    """Return copy of img with red bbox drawn."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def load_tracks_by_id(seq, cls):
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
    import random
    rng = random.Random(seed)
    cands = []
    for tid, frames in tracks_by_id.items():
        fids = sorted(frames.keys())
        for f in fids:
            if (f - delta) in frames and (f + delta) in frames:
                cands.append((tid, f))
    rng.shuffle(cands)
    return cands[:n_samples]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="0011")
    p.add_argument("--cls", default="car")
    p.add_argument("--expr", default="moving cars")
    p.add_argument("--n_calls", type=int, default=30)
    p.add_argument("--delta", type=int, default=10)
    p.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--max_dim", type=int, default=560)
    args = p.parse_args()

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    print("Loading model...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb, device_map="auto"
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model_id)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    tracks_by_id = load_tracks_by_id(args.seq, args.cls)
    triplets = sample_triplets(tracks_by_id, args.delta, args.n_calls)
    print(f"Triplets: {len(triplets)}", flush=True)

    prompt = (
        f"These three images show the SAME scene at three moments (Δ={args.delta} frames). "
        f"The red box highlights ONE specific object across the three frames. "
        f"Judge whether this object's motion matches: \"{args.expr}\". "
        f"Score 0 (clearly not matching) to 10 (clearly matching). "
        f"Output ONE integer 0-10 only."
    )

    img_dir = os.path.join(KITTI_ROOT, args.seq)
    scores = []
    t0 = time.time()
    for i, (tid, fc) in enumerate(triplets):
        imgs = []
        ok = True
        for f in [fc - args.delta, fc, fc + args.delta]:
            img_path = os.path.join(img_dir, f"{f - 1:06d}.png")
            if not os.path.exists(img_path):
                ok = False
                break
            img = Image.open(img_path).convert("RGB")
            img = draw_bbox(img, tracks_by_id[tid][f])
            img.thumbnail((args.max_dim, args.max_dim))
            imgs.append(img)
        if not ok:
            continue

        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": im} for im in imgs],
            {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        resp = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        s = parse_score(resp)
        if s is not None:
            scores.append(s)
        if i < 8 or i % 10 == 0:
            print(f"  [{i+1}] tid={tid} fc={fc} resp='{resp}' score={s}", flush=True)

    elapsed = time.time() - t0
    n = len(triplets)
    print()
    print(f"=== Overlay result ({args.expr}, Δ={args.delta}) ===")
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
