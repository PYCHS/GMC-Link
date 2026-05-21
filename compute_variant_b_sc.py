"""Variant B sc derivation: sc = std(model_logit) / std(raw_cos) per arch per axis.

Uses sw raw_cos caches (ship aligner) + existing iKUN/FH prediction sources.
Reports derived sc values per arch + axis for downstream HOTA validation.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np

from gmc_link.demo_inference import load_ikun_scores

# ─── Paths ───
CASCADE_FULL = "/home/seanachan/GMC-Link/iKUN/ikun_results_v1_cascade_full.json"
FLEXHOOK_V1  = "/home/seanachan/FlexHook/retest-kitti-1/refer-kitti-best/result_0.json"
FLEXHOOK_V2  = "/home/seanachan/FlexHook/retest-kitti-2/refer-kitti-v2-best/result_0.json"

V1_SEQS = ["0005", "0011", "0013"]
V2_SEQS = ["0005", "0011", "0013", "0019"]
V1_EXPR_DIR = "/home/seanachan/GMC-Link/refer-kitti/expression"
V2_EXPR_DIR = "/home/seanachan/data/Dataset/refer-kitti-v2/expression"
SEED = 0  # use single seed for sc derivation (std stable cross-seed)

MOTION_KW = ["moving","walking","running","turning","faster","slower","braking",
             "drive","driving","speed","speeding","walk","run","jog","jogging"]
STATIC_KW = ["parking","parked","stopped","stop","stand","static","stationary"]
def is_motion(e): return any(k in e.lower() for k in MOTION_KW)


def collect_pairs(ikun_scores_dict, gmc_cache, expr_list):
    """Walk all (expr, fid, oid) tuples; return motion + appear lists of (cs, gmc) pairs."""
    motion_pairs, appear_pairs = [], []
    for expr in expr_list:
        if expr not in gmc_cache:
            continue
        # cs lookup function depends on source
        cs_lookup = ikun_scores_dict.get(expr)  # may be a dict or None
        if cs_lookup is None:
            continue
        gmc_expr = gmc_cache[expr]
        target = motion_pairs if is_motion(expr) else appear_pairs
        for fid_str, frame_dict in gmc_expr.items():
            fid = int(fid_str)
            cs_frame = cs_lookup.get(fid, {})
            for oid_str, gmc_val in frame_dict.items():
                oid = int(oid_str)
                cs = cs_frame.get(oid)
                if cs is None: continue
                target.append((float(cs), float(gmc_val)))
    return motion_pairs, appear_pairs


def std_report(name, pairs):
    if not pairs:
        return None
    arr = np.array(pairs)
    std_cs = arr[:,0].std()
    std_gmc = arr[:,1].std()
    sc = std_cs / std_gmc if std_gmc > 0 else float('nan')
    print(f"  {name}: n={len(pairs):6d}  std(cs)={std_cs:.4f}  std(gmc)={std_gmc:.4f}  → sc={sc:.4f}")
    return {"n": len(pairs), "std_cs": std_cs, "std_gmc": std_gmc, "sc": sc}


# ─── iKUN ───
print("=" * 60)
print("iKUN (cascade scores + sw raw_cos)")
print("=" * 60)
ikun_motion_all, ikun_appear_all = [], []
for seq in V1_SEQS:
    gmc_path = f"gmc_link/gmc_scores_v1_{seq}_sharedweight_seed{SEED}_rawcos_cache.json"
    if not os.path.exists(gmc_path):
        print(f"  [skip] missing {gmc_path}")
        continue
    with open(gmc_path) as f:
        gmc_cache = json.load(f)
    exprs = list(gmc_cache.keys())
    # iKUN cs is per-(seq, expr) → load lazily
    cs_dict = {}
    for expr in exprs:
        s = load_ikun_scores(CASCADE_FULL, seq, expr)
        cs_dict[expr] = s
    mp, ap = collect_pairs(cs_dict, gmc_cache, exprs)
    ikun_motion_all.extend(mp)
    ikun_appear_all.extend(ap)
    print(f"  seq {seq}: motion={len(mp)} appear={len(ap)}")

print("\niKUN derived sc:")
ikun_mot = std_report("MOTION", ikun_motion_all)
ikun_app = std_report("APPEAR", ikun_appear_all)


# ─── FlexHook V1 ───
print("\n" + "=" * 60)
print("FlexHook V1 (margin = score[1] - score[0] + sw raw_cos)")
print("=" * 60)

_FH_CACHE = {}
def load_fh_results(path):
    if path not in _FH_CACHE:
        with open(path) as f:
            _FH_CACHE[path] = json.load(f)
    return _FH_CACHE[path]

def load_fh_scores(results_path, seq, expr_text):
    """FH result_0.json: video → frame_id → obj_id → expr_text → [s0, s1]; return {fid: {oid: margin}}."""
    all_results = load_fh_results(results_path)
    video_dict = all_results.get(seq, {})
    scores = defaultdict(dict)
    for fid_str, frame_dict in video_dict.items():
        fid = int(fid_str)
        for oid_str, expr_dict in frame_dict.items():
            oid = int(oid_str)
            raw = expr_dict.get(expr_text)
            if raw is not None and isinstance(raw, list) and len(raw) == 2:
                scores[fid][oid] = float(raw[1] - raw[0])
    return scores

def collect_fh_pairs(results_path, expr_dir_root, seq, gmc_cache):
    """FH-specific: GMC keyed by expr_id (hyphenated); results keyed by sentence text."""
    expr_dir = os.path.join(expr_dir_root, seq)
    motion_pairs, appear_pairs = [], []
    fh_all = load_fh_results(results_path)
    seq_dict = fh_all.get(seq, {})
    for expr_id in gmc_cache.keys():
        json_path = os.path.join(expr_dir, f"{expr_id}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            sentence = json.load(f).get("sentence")
        if not sentence:
            continue
        gmc_expr = gmc_cache[expr_id]
        target = motion_pairs if is_motion(expr_id) else appear_pairs
        for fid_str, frame_dict in gmc_expr.items():
            seq_frame = seq_dict.get(fid_str)
            if seq_frame is None: continue
            for oid_str, gmc_val in frame_dict.items():
                obj_dict = seq_frame.get(oid_str)
                if obj_dict is None: continue
                raw = obj_dict.get(sentence)
                if raw is None or not isinstance(raw, list) or len(raw) != 2:
                    continue
                margin = float(raw[1] - raw[0])
                target.append((margin, float(gmc_val)))
    return motion_pairs, appear_pairs

if not os.path.exists(FLEXHOOK_V1):
    print(f"  [skip] missing {FLEXHOOK_V1}")
else:
    fhv1_motion_all, fhv1_appear_all = [], []
    for seq in V1_SEQS:
        gmc_path = f"gmc_link/gmc_scores_flexhook_v1_{seq}_sharedweight_seed{SEED}_rawcos_cache.json"
        if not os.path.exists(gmc_path):
            print(f"  [skip] missing {gmc_path}")
            continue
        with open(gmc_path) as f:
            gmc_cache = json.load(f)
        mp, ap = collect_fh_pairs(FLEXHOOK_V1, V1_EXPR_DIR, seq, gmc_cache)
        fhv1_motion_all.extend(mp)
        fhv1_appear_all.extend(ap)
        print(f"  seq {seq}: motion={len(mp)} appear={len(ap)}")

    print("\nFH V1 derived sc:")
    fhv1_mot = std_report("MOTION", fhv1_motion_all)
    fhv1_app = std_report("APPEAR", fhv1_appear_all)


# ─── FlexHook V2 ───
print("\n" + "=" * 60)
print("FlexHook V2 (margin + sw raw_cos)")
print("=" * 60)
if not os.path.exists(FLEXHOOK_V2):
    print(f"  [skip] missing {FLEXHOOK_V2}")
else:
    fhv2_motion_all, fhv2_appear_all = [], []
    for seq in V2_SEQS:
        gmc_path = f"gmc_link/gmc_scores_flexhook_v2_raw_{seq}_sharedweight_seed{SEED}_rawcos_cache.json"
        if not os.path.exists(gmc_path):
            print(f"  [skip] missing {gmc_path}")
            continue
        with open(gmc_path) as f:
            gmc_cache = json.load(f)
        mp, ap = collect_fh_pairs(FLEXHOOK_V2, V2_EXPR_DIR, seq, gmc_cache)
        fhv2_motion_all.extend(mp)
        fhv2_appear_all.extend(ap)
        print(f"  seq {seq}: motion={len(mp)} appear={len(ap)}")

    print("\nFH V2 derived sc:")
    fhv2_mot = std_report("MOTION", fhv2_motion_all)
    fhv2_app = std_report("APPEAR", fhv2_appear_all)


# ─── Summary table ───
print("\n" + "=" * 60)
print("SUMMARY — derived sc vs hand-tuned ship sc")
print("=" * 60)
print(f"  {'arch':<8} {'axis':<8} {'derived':>10} {'ship':>10} {'ratio':>10}")
SHIP = {
    ("iKUN",  "MOTION"): 0.9,
    ("iKUN",  "APPEAR"): 0.30,
    ("FH V1", "MOTION"): 10.0,
    ("FH V1", "APPEAR"): 3.5,
    ("FH V2", "MOTION"): 10.0,
    ("FH V2", "APPEAR"): 3.5,
}
for arch, key in [("iKUN", "ikun"), ("FH V1", "fhv1"), ("FH V2", "fhv2")]:
    mot = locals().get(f"{key}_mot")
    app = locals().get(f"{key}_app")
    for axis_name, info in [("MOTION", mot), ("APPEAR", app)]:
        if info is None: continue
        ship_sc = SHIP[(arch, axis_name)]
        ratio = info["sc"] / ship_sc if ship_sc > 0 else float('nan')
        print(f"  {arch:<8} {axis_name:<8} {info['sc']:>10.4f} {ship_sc:>10.4f} {ratio:>10.3f}")
