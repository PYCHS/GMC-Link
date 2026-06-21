"""Lever B Phase B1: dual-cascade (what + where) fused score → ship recipe HOTA.

cs_fused[fid][oid] = w_what * mean(logits_what) + w_where * mean(logits_where)

Then applies the LOCKED iKUN linear-additive ship recipe:
    MOVING/OTHER : fused = cs_fused + b + 1.0*(gmc-0.5)*0.9 ; keep if > +0.17
    APPEAR/STATIC: fused = cs_fused + b + 1.0*(gmc-0.5)*0.30; keep if > +0.10

Reference: ship single-seed = 44.586, multi-seed (n=3) = 44.608 ± 0.024.

Sweep w_what ∈ {0.3, 0.5, 0.7}, w_where = 1 - w_what. Gate B1:
    POS Δship ≥ +0.10 → proceed to B2 multi-seed.
    NEG Δship <  +0.05 → kill Lever B.

Usage:
    python run_whatwhere_b1_eval.py \
        --cache_what  iKUN/ikun_results_v1_cascade_what.json \
        --cache_where iKUN/ikun_results_v1_cascade_where.json \
        --grid 0.3 0.5 0.7
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/seanachan/GMC-Link")
sys.path.insert(0, "/home/seanachan/iKUN")

from gmc_link.demo_inference import load_neuralsort_tracks
from utils import expression_conversion as ikun_expression_conversion


DATA_ROOT      = "/home/seanachan/GMC-Link/refer-kitti"
TRACK_DIR      = "/home/seanachan/GMC-Link/NeuralSORT"
GT_TEMPLATE    = "/home/seanachan/data/Dataset/refer-kitti/gt_template_old"
TEXT_FEAT_JSON = "/home/seanachan/GMC-Link/iKUN/text_feat_bboxNum_v1.json"
_GMC_SUFFIX    = os.environ.get("GMC_SUFFIX", "")
GMC_CACHE_TPL  = "/home/seanachan/GMC-Link/gmc_link/gmc_scores_v1_{seq}" + _GMC_SUFFIX + "_cache.json"
TRACKEVAL      = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"
OUT_ROOT       = "/home/seanachan/GMC-Link/hota_eval_whatwhere_b1"

TEST_SEQS = ["0005", "0011", "0013"]
FRAMES = {"0005": (0, 296), "0011": (0, 372), "0013": (0, 339)}
SIM_A, SIM_B, SIM_TAU = 8.0, -0.1, 100.0

# Ship recipe (locked, paper-canonical)
ALPHA_M, SCALE_M, THR_M = 1.0, 0.9, 0.17
ALPHA_A, SCALE_A, THR_A = 1.0, 0.30, 0.10

MOTION_KW = ["moving","walking","running","turning","faster","slower","braking",
             "parking","parked","stopped","stop","stand","static","stationary","accelerat"]
STATIC_KW = ["parking","parked","stopped","stop","stand","static","stationary"]


def is_motion(e): return any(k in e.lower() for k in MOTION_KW)
def classify(e):
    if not is_motion(e): return "APPEARANCE"
    if any(k in e.lower() for k in STATIC_KW): return "STATIC"
    return "MOVING"


def compute_simcalib_bias(text_feat, exprs):
    train_dict, test_dict = text_feat["train"], text_feat["test"]
    keys = list(train_dict.keys())
    FEATS = np.array([train_dict[k]["feature"] for k in keys])
    PROBS = np.array([train_dict[k]["probability"] for k in keys])
    bias = {}
    for expr in exprs:
        en = ikun_expression_conversion(expr)
        target = test_dict if en in test_dict else train_dict
        if en not in target: bias[expr] = 0.0; continue
        feat = np.array(target[en]["feature"])[None, :]
        sim = (feat @ FEATS.T)[0]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
        w = np.exp(SIM_TAU * sim); w = w / w.sum()
        prob = float((w * PROBS).sum())
        bias[expr] = SIM_A * prob + SIM_B
    return bias


def merged_ns(seq):
    car = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "car", "predict.txt"))
    ped = load_neuralsort_tracks(os.path.join(TRACK_DIR, seq, "pedestrian", "predict.txt"))
    max_car = 0
    for fid, dets in car.items():
        for oid, *_ in dets: max_car = max(max_car, oid)
    ns = defaultdict(list)
    for fid, dets in car.items(): ns[fid].extend(dets)
    for fid, dets in ped.items():
        ns[fid].extend([(oid+max_car, x, y, w, h) for oid, x, y, w, h in dets])
    return ns


def load_cascade_means(path):
    """Load {video → {oid → {fid → {expr_raw → list[logit]}}}} → MEAN-per-(fid,oid,expr).
    Output: {video: {expr_raw: {fid: {oid: mean_logit}}}}"""
    raw = json.load(open(path))
    out = {}
    for video, obj_dict in raw.items():
        out_v = defaultdict(lambda: defaultdict(dict))
        for oid_str, frame_dict in obj_dict.items():
            oid = int(oid_str)
            for fid_str, expr_dict in frame_dict.items():
                fid = int(fid_str)
                for expr, logits in expr_dict.items():
                    out_v[expr][fid][oid] = float(np.mean(logits))
        out[video] = out_v
    return out


def gen_predicts(text_feat, gmc_caches, cascade_what, cascade_where,
                 w_what, w_where, run_dir):
    res_dir = os.path.join(run_dir, "results")
    if os.path.exists(res_dir): shutil.rmtree(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    seqmap_lines = []

    for seq in TEST_SEQS:
        ns = merged_ns(seq)
        expr_dir = os.path.join(DATA_ROOT, "expression", seq)
        exprs = sorted(f.replace(".json", "") for f in os.listdir(expr_dir) if f.endswith(".json"))
        bias = compute_simcalib_bias(text_feat, exprs)
        gmc_seq = gmc_caches.get(seq, {})
        min_f, max_f = FRAMES[seq]
        cw = cascade_what.get(seq, {})
        cwhe = cascade_where.get(seq, {})

        for expr in exprs:
            outd = os.path.join(res_dir, seq, expr); os.makedirs(outd, exist_ok=True)
            gt_src = os.path.join(GT_TEMPLATE, seq, expr, "gt.txt")
            gt_dst = os.path.join(outd, "gt.txt")
            if os.path.exists(gt_src): shutil.copy2(gt_src, gt_dst)
            else: open(gt_dst, "w").close()
            seqmap_lines.append(f"{seq}+{expr}")

            b = bias.get(expr, 0.0)
            motion = is_motion(expr)
            per_expr_gmc = gmc_seq.get(expr, {})
            cw_expr = cw.get(expr, {})
            cwh_expr = cwhe.get(expr, {})

            rows = []
            for fid, dets in ns.items():
                if not (min_f < fid < max_f): continue
                for oid, x, y, w, h in dets:
                    cs_w = cw_expr.get(fid, {}).get(oid)
                    cs_h = cwh_expr.get(fid, {}).get(oid)
                    if cs_w is None and cs_h is None: continue
                    if cs_w is None: cs_fused = cs_h
                    elif cs_h is None: cs_fused = cs_w
                    else: cs_fused = w_what * cs_w + w_where * cs_h

                    if motion:
                        gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), 0.5))
                        fused = cs_fused + b + ALPHA_M * (gmc - 0.5) * SCALE_M
                        thr = THR_M
                    else:
                        gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), 0.5))
                        fused = cs_fused + b + ALPHA_A * (gmc - 0.5) * SCALE_A
                        thr = THR_A
                    if fused > thr:
                        rows.append((fid, oid, x, y, w, h))

            with open(os.path.join(outd, "predict.txt"), "w") as f:
                for fid, oid, x, y, w, h in rows:
                    f.write(f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")

    sm = os.path.join(run_dir, "seqmap.txt")
    open(sm, "w").write("\n".join(seqmap_lines) + "\n")
    return res_dir, sm


def run_te(seqmap_path, results_dir, class_filter=None):
    if class_filter is None:
        sm = seqmap_path
    else:
        sm = os.path.join(os.path.dirname(seqmap_path), f"seqmap_{class_filter}.txt")
        lines = [l for l in open(seqmap_path).read().splitlines()
                 if l and classify(l.split("+", 1)[1]) == class_filter]
        if not lines: return None
        open(sm, "w").write("\n".join(lines) + "\n")
    sp = os.path.join(results_dir, "pedestrian_summary.txt")
    if os.path.exists(sp): os.remove(sp)
    cmd = [sys.executable, TRACKEVAL,
           "--METRICS", "HOTA",
           "--SEQMAP_FILE", os.path.abspath(sm),
           "--SKIP_SPLIT_FOL", "True",
           "--GT_FOLDER", os.path.abspath(results_dir),
           "--TRACKERS_FOLDER", os.path.abspath(results_dir),
           "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
           "--TRACKERS_TO_EVAL", os.path.abspath(results_dir),
           "--USE_PARALLEL", "False", "--PLOT_CURVES", "False", "--PRINT_CONFIG", "False"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(TRACKEVAL))
    if not os.path.exists(sp):
        sys.stderr.write(f"FAIL ({class_filter}) rc={proc.returncode}\n{proc.stderr[-1500:]}\n")
        return None
    return float(open(sp).read().splitlines()[1].split()[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_what", required=True)
    p.add_argument("--cache_where", required=True)
    p.add_argument("--grid", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = p.parse_args()

    print("Loading caches + simcalib refs + GMC caches...", flush=True)
    text_feat = json.load(open(TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(GMC_CACHE_TPL.format(seq=s))) for s in TEST_SEQS}

    print(f"Loading cascade-what:  {args.cache_what}", flush=True)
    cw = load_cascade_means(args.cache_what)
    print(f"Loading cascade-where: {args.cache_where}", flush=True)
    cwhe = load_cascade_means(args.cache_where)

    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    for w_what in args.grid:
        w_where = 1.0 - w_what
        tag = f"ww{w_what:.2f}_wh{w_where:.2f}"
        run_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== {tag} ===", flush=True)
        res_dir, sm = gen_predicts(text_feat, gmc_caches, cw, cwhe,
                                    w_what, w_where, run_dir)
        pooled = run_te(sm, res_dir)
        moving = run_te(sm, res_dir, class_filter="MOVING")
        static = run_te(sm, res_dir, class_filter="STATIC")
        appear = run_te(sm, res_dir, class_filter="APPEARANCE")
        rows.append((tag, w_what, w_where, pooled, appear, moving, static))
        print(f"  pooled={pooled}  APPEAR={appear}  MOVING={moving}  STATIC={static}",
              flush=True)

    print("\n=== Lever B Phase B1: what/where dual cosine sweep ===")
    print("tag             w_what  w_where  pooled   APPEAR   MOVING   STATIC")
    for tag, ww, wh, p_, ap, mo, st in rows:
        print(f"{tag:<15} {ww:5.2f}   {wh:5.2f}    "
              f"{p_:.3f}   {ap:.3f}   {mo:.3f}   {st:.3f}")
    print("\nShip seed-1 ref:         44.586")
    print("Ship multi-seed (n=3):   44.608 ± 0.024")
    print("Gate B1: Δ≥+0.10 → POS proceed multi-seed; Δ<+0.05 → KILL")


if __name__ == "__main__":
    main()
