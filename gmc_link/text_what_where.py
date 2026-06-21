"""spaCy POS decomposition of RMOT expressions into what/where tokens.

What = noun phrase + appearance attributes (NOUN, PROPN, ADJ + color/material).
Where = motion + spatial relations (VERB, ADV, motion-keywords, spatial-keywords).

Used by Lever B (CDRMOT-inspired dual cosine) to drive two cascade passes whose
logits are linearly combined: cs_fused = w_what*cs_what + w_where*cs_where.

Parses both train + test expression sets from refer-kitti V1 and writes a single
mapping JSON: expr_raw → {what, where}.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import spacy

sys.path.insert(0, "/home/seanachan/iKUN")
from utils import expression_conversion as ikun_expression_conversion


DATA_ROOT = "/home/seanachan/GMC-Link/refer-kitti"
TEXT_FEAT_JSON = "/home/seanachan/GMC-Link/iKUN/text_feat_bboxNum_v1.json"
DEFAULT_OUT = "/home/seanachan/GMC-Link/iKUN/expr_what_where_v1.json"

# Motion keywords: anything moving/static/turning → where (regardless of POS)
MOTION_KW = {
    "moving", "walking", "running", "turning", "faster", "slower",
    "braking", "accelerat", "parking", "parked", "stopped", "stop",
    "stand", "static", "stationary", "heading", "going", "traveling",
    "drive", "driving", "drove", "ride", "riding",
}

# Spatial / relation keywords: place / direction descriptors → where
SPATIAL_KW = {
    "left", "right", "front", "back", "behind", "side", "ahead",
    "near", "away", "beside", "above", "below", "same", "opposite",
    "counter", "direction", "ours", "us",
}

# Color/material keywords: clearly appearance → what (override POS if needed)
COLOR_KW = {
    "red", "blue", "green", "white", "black", "silver", "gray", "grey",
    "yellow", "orange", "purple", "brown", "pink", "gold",
    "light-color", "light", "dark", "pale", "bright", "hue", "color",
}


def _classify_token(tok):
    """Return 'what' | 'where' | 'drop' for a single spaCy token."""
    word = tok.text.lower().strip("-")
    # priority: explicit lexical override > POS
    if any(kw in word for kw in MOTION_KW):
        return "where"
    if word in SPATIAL_KW:
        return "where"
    if word in COLOR_KW or any(c in word for c in COLOR_KW):
        return "what"
    if tok.pos_ in {"VERB", "ADV"}:
        return "where"
    if tok.pos_ in {"ADP"}:  # prepositions: "in", "on", "of" → relational
        return "where"
    if tok.pos_ in {"NOUN", "PROPN", "ADJ"}:
        return "what"
    # DET, PUNCT, AUX, SCONJ, CCONJ → drop
    return "drop"


def parse_expression(text, nlp):
    """Return (what_str, where_str). Falls back to full text if either is empty."""
    norm = text.replace("-", " ")
    doc = nlp(norm)
    what_toks, where_toks = [], []
    for tok in doc:
        cls = _classify_token(tok)
        if cls == "what":
            what_toks.append(tok.text)
        elif cls == "where":
            where_toks.append(tok.text)
    what_str = " ".join(what_toks).strip() or norm
    where_str = " ".join(where_toks).strip() or norm
    return what_str, where_str


def collect_v1_expressions():
    """Enumerate every raw expression file across the V1 test sequences AND every
    train/test key from the canonical text_feat JSON (iKUN-converted form).

    Returns a set of (raw_form, converted_form) tuples — the cascade
    dataloader yields `expression_raw=expression` (file stem) and
    `expression_new=ikun_expression_conversion(expression)`. We must parse
    *expression_new* (the form the model sees) but key the output by
    *expression_raw* (the runtime lookup key).
    """
    pairs = set()
    expr_root = os.path.join(DATA_ROOT, "expression")
    for seq in sorted(os.listdir(expr_root)):
        seq_dir = os.path.join(expr_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for fn in os.listdir(seq_dir):
            if not fn.endswith(".json"):
                continue
            raw = fn[:-5]
            pairs.add((raw, ikun_expression_conversion(raw)))

    txtf = json.load(open(TEXT_FEAT_JSON))
    for split in ("train", "test"):
        for key in txtf[split].keys():
            pairs.add((key, key))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--inspect", type=int, default=0,
                    help="Print N random parses for sanity.")
    args = ap.parse_args()

    print("Loading spaCy en_core_web_sm...", flush=True)
    nlp = spacy.load("en_core_web_sm")

    pairs = collect_v1_expressions()
    print(f"Collected {len(pairs)} unique expressions.", flush=True)

    out = {}
    for raw, converted in sorted(pairs):
        what, where = parse_expression(converted, nlp)
        out[raw] = {
            "converted": converted,
            "what": what,
            "where": where,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} (n={len(out)})", flush=True)

    if args.inspect:
        import random
        keys = random.sample(list(out.keys()), min(args.inspect, len(out)))
        for k in keys:
            v = out[k]
            print(f"  raw={k!r}")
            print(f"    converted={v['converted']!r}")
            print(f"    what     ={v['what']!r}")
            print(f"    where    ={v['where']!r}")

    # Coverage stats
    n_both = sum(1 for v in out.values()
                 if v["what"] != v["converted"] and v["where"] != v["converted"])
    n_what_only = sum(1 for v in out.values()
                      if v["what"] != v["converted"] and v["where"] == v["converted"])
    n_where_only = sum(1 for v in out.values()
                       if v["what"] == v["converted"] and v["where"] != v["converted"])
    n_fallback = sum(1 for v in out.values()
                     if v["what"] == v["converted"] and v["where"] == v["converted"])
    print(f"  split stats: both-decomposed={n_both} what-only={n_what_only} "
          f"where-only={n_where_only} fallback-both-same={n_fallback}")


if __name__ == "__main__":
    main()
