"""C1 calibration probe: 100-call test on Qwen2-VL-2B.

Verifies: (a) model loads under 3GB GPU pressure, (b) prompt returns parseable
0-10 integer ≥90% of time, (c) latency ≤2s/call.

Uses single seq (0011) + single motion expression for fixed scope.
Crops bboxes from NeuralSORT predict.txt, queries LVLM, parses score.
"""
import argparse
import os
import re
import time

import torch
from PIL import Image


KITTI_ROOT = "/home/seanachan/GMC-Link/refer-kitti/KITTI/training/image_02"
NS_ROOT = "/home/seanachan/GMC-Link/NeuralSORT"


def parse_score(text):
    """Extract first integer 0-10 from response text. Returns None if absent."""
    m = re.search(r"\b([0-9]|10)\b", text)
    if m is None:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 10 else None


def crop_box(img, box, pad_frac=0.20):
    """Crop xyxy bbox from PIL image with fractional padding."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = pad_frac * w, pad_frac * h
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(img.width, int(x2 + px))
    y2p = min(img.height, int(y2 + py))
    return img.crop((x1p, y1p, x2p, y2p))


def load_random_n_tracks(seq, cls, n, seed=0):
    """Load N random (fid, x1y1x2y2) entries spread across NS predict.txt."""
    import random
    path = os.path.join(NS_ROOT, seq, cls, "predict.txt")
    all_rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            all_rows.append((fid, (x, y, x + w, y + h)))
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    return all_rows[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="0011")
    p.add_argument("--cls", default="car")
    p.add_argument("--expr", default="moving-cars")
    p.add_argument("--n_calls", type=int, default=100)
    p.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--use_int4", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt_variant", default="score", choices=["score", "yesno"])
    args = p.parse_args()

    print(f"Loading {args.model_id} (int4={args.use_int4})...", flush=True)
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    kwargs = {}
    if args.use_int4:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = args.device

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, **kwargs)
    proc = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    print(f"Model loaded. GPU mem alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    expr_clean = args.expr.replace("-", " ")
    if args.prompt_variant == "score":
        prompt = (
            f"Score from 0 to 10 how well this region matches the description: "
            f"\"{expr_clean}\". Respond with only a single integer between 0 and 10."
        )
    elif args.prompt_variant == "yesno":
        prompt = (
            f"Does this region show {expr_clean}? "
            f"Score 0 for clearly no, 10 for clearly yes, intermediate for uncertain. "
            f"Respond with a single integer 0 to 10."
        )
    else:
        raise ValueError(args.prompt_variant)

    tracks = load_random_n_tracks(args.seq, args.cls, args.n_calls)
    print(f"Loaded {len(tracks)} tracks from {args.seq}/{args.cls}", flush=True)

    img_dir = os.path.join(KITTI_ROOT, args.seq)
    parsed_ok = 0
    scores = []
    t0 = time.time()
    for i, (fid, box) in enumerate(tracks):
        fname = f"{fid - 1:06d}.png"  # NS predict 1-indexed → 0-indexed file
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        crop = crop_box(img, box)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[crop], padding=True, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        resp = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        score = parse_score(resp)
        if score is not None:
            parsed_ok += 1
            scores.append(score)
        if i < 5 or i % 25 == 0:
            print(f"  [{i+1}/{len(tracks)}] fid={fid} resp='{resp}' score={score}", flush=True)

    elapsed = time.time() - t0
    n = len(tracks)
    print()
    print(f"=== Calibration result ===")
    print(f"calls: {n}, parsed: {parsed_ok} ({100*parsed_ok/n:.1f}%)")
    print(f"latency: {elapsed/n:.2f} s/call, total {elapsed:.1f} s")
    if scores:
        from collections import Counter
        h = Counter(scores)
        print(f"score histogram: {dict(sorted(h.items()))}")
        print(f"mean={sum(scores)/len(scores):.2f}, n_unique={len(h)}")
    gpu_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU peak: {gpu_peak:.2f} GB")


if __name__ == "__main__":
    main()
