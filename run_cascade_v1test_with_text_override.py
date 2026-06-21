"""Cascade KUM inference on V1 test seqs (0005, 0011, 0013) with text override.

Generic driver: rebinds `expression_new` per batch via a precomputed mapping
{expression_raw: substituted_text} so the cascade text encoder sees the
substituted form. Used by Lever B (what/where dual cosine) to produce two
parallel cascade-score caches.

Output JSON schema matches `iKUN/ikun_results_v1_cascade_full.json`:
    {video: {oid: {fid: {expr_raw: [logit, ...]}}}}

Usage:
    python run_cascade_v1test_with_text_override.py \
        --mapping iKUN/expr_what_where_v1.json --field what \
        --out iKUN/ikun_results_v1_cascade_what.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm


def multi_dim_dict(n, leaf_factory):
    if n == 1:
        return defaultdict(leaf_factory)
    return defaultdict(lambda n=n - 1, lf=leaf_factory: multi_dim_dict(n, lf))


def to_plain(d):
    """Recursively convert defaultdict → dict for json.dump."""
    if isinstance(d, defaultdict):
        return {k: to_plain(v) for k, v in d.items()}
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", required=True,
                   help="JSON file mapping expression_raw → {field: text, ...}")
    p.add_argument("--field", required=True,
                   help="Which key inside each mapping value to use as substituted text.")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default="iKUN_cascade_attention.pth")
    args = p.parse_args()

    if os.path.exists(args.out):
        print(f"Already exists, skipping: {args.out}")
        return

    sys.argv = [
        "test.py",
        "--save_root", "/home/seanachan/GMC-Link",
        "--exp_name", "iKUN",
        "--kum_mode", "cascade attention",
        "--test_ckpt", args.ckpt,
        "--test_bs", "1",
        "--num_workers", "4",
    ]
    sys.path.insert(0, "/home/seanachan/iKUN")

    import utils as ikun_utils
    # Default VIDEOS['test'] = ['0005','0011','0013']; keep as-is.

    from opts import opt
    from utils import load_from_ckpt
    from model import get_model
    from dataloader import get_dataloader

    # Load text override mapping
    text_map = json.load(open(args.mapping))
    miss_log = set()

    def substitute(expr_raw_list):
        out = []
        for raw in expr_raw_list:
            v = text_map.get(raw)
            if v is None or args.field not in v:
                miss_log.add(raw)
                out.append(raw)
            else:
                out.append(v[args.field])
        return out

    import clip

    def tokenize(texts):
        return clip.tokenize(texts, truncate=True)

    print(f"Test seqs: {ikun_utils.VIDEOS['test']}", flush=True)
    model = get_model(opt, "Model")
    ckpt_path = os.path.join(opt.save_root, opt.test_ckpt)
    print(f"Loading ckpt: {ckpt_path}", flush=True)
    model, _ = load_from_ckpt(model, ckpt_path)
    model.cuda().eval()

    dl = get_dataloader("test", opt, "Track_Dataset")
    print(f"Batches: {len(dl)}", flush=True)

    OUTPUTS = multi_dim_dict(4, list)
    n_batches = 0
    n_substitutions = 0
    n_identical = 0
    with torch.no_grad():
        for data in tqdm(dl, desc=f"cascade-{args.field}"):
            sub_text = substitute(data["expression_raw"])
            for orig, sub in zip(data["expression_new"], sub_text):
                if orig == sub:
                    n_identical += 1
                else:
                    n_substitutions += 1
            inputs = dict(
                local_img=data["cropped_images"].cuda(),
                global_img=data["global_images"].cuda(),
                exp=tokenize(sub_text).cuda(),
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                similarity = model(inputs)["logits"].cpu()
            for idx in range(len(data["video"])):
                for fid in range(int(data["start_frame"][idx]),
                                 int(data["stop_frame"][idx]) + 1):
                    frame_dict = OUTPUTS[data["video"][idx]][int(data["obj_id"][idx])][int(fid)]
                    frame_dict[data["expression_raw"][idx]].append(
                        similarity[idx].cpu().numpy().tolist()
                    )
            n_batches += 1

    if miss_log:
        print(f"WARN: {len(miss_log)} expressions had no mapping (fallback to raw):",
              flush=True)
        for k in list(miss_log)[:10]:
            print(f"  {k!r}", flush=True)

    print(f"Substitutions={n_substitutions} identical={n_identical}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(to_plain(OUTPUTS), open(args.out, "w"))
    print(f"Wrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
