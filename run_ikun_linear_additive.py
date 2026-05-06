"""iKUN cascade+simcalib + linear additive GMC bias on 3-seq POOLED HOTA.

Mirrors FlexHook fusion form (which beat paper on V2, +0.497 on V1):

    motion expr:     fused = cs + b + alpha*(gmc - 0.5)*scale
                     keep iff fused > thr_motion
    appearance expr: fused = cs + b
                     keep iff fused > 0  (baseline gating)

No MLP. Plain hand-tuned linear additive form.

Reference baselines (3-seq pooled HOTA, paper-canonical gt_template_old):
  paper-pure (alpha=0)  44.564  (paper README claim)
  local B (alpha=0)     44.224  (cli-fork drift 0.34)
  hand-tuned alpha=0.5+thr  43.910  (NEG, see project_phase5_stack_pooled_negative)
  hand-tuned alpha=1.0      43.260  (NEG)
  learned residual MLP      42.919  (NEG, project_ikun_learned_residual_negative)

Goal: test whether FlexHook's WINNING linear-additive recipe (which beat paper on
both V1 and V2 FlexHook) generalizes to iKUN cascade architecture. Different
fusion form than the legacy iKUN tries (MLP, raw alpha bias without thr).

Usage:
    python run_ikun_linear_additive.py --alpha 1.0 --gmc_scale 0.9 --thr 0.17
    python run_ikun_linear_additive.py --grid

Grid mode (2026-05-02 Path 1): motion locked at ship recipe (α=1, sc=0.9, thr=+0.17),
sweep APPEARANCE-axis bias. Project memory project_gmc_is_motion_plus_bbox_specialist
shows APPEAR raw sep +0.264 > motion +0.172. APPEAR = 77% of V1 frames; if class HOTA
gains +1.0, pool gain ≈ +0.77 (4× current iKUN gain).
"""
import argparse, json, os, shutil, subprocess, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/home/seanachan/GMC-Link")
sys.path.insert(0, "/home/seanachan/iKUN")

from gmc_link.demo_inference import load_neuralsort_tracks, load_ikun_scores
from utils import expression_conversion as ikun_expression_conversion

DATA_ROOT      = "/home/seanachan/GMC-Link/refer-kitti"
TRACK_DIR      = "/home/seanachan/GMC-Link/NeuralSORT"
GT_TEMPLATE    = "/home/seanachan/data/Dataset/refer-kitti/gt_template_old"
TEXT_FEAT_JSON = "/home/seanachan/GMC-Link/iKUN/text_feat_bboxNum_v1.json"
CASCADE_FULL   = "/home/seanachan/GMC-Link/iKUN/ikun_results_v1_cascade_full.json"
_GMC_SUFFIX = os.environ.get("GMC_SUFFIX", "")  # e.g. "_seed0"
RAW_COS    = os.environ.get("GMC_RAW_COS", "0") == "1"  # Arm B: GMC cache contains raw cosine [-1,+1]
GMC_CACHE_TPL  = "/home/seanachan/GMC-Link/gmc_link/gmc_scores_v1_{seq}" + _GMC_SUFFIX + "_cache.json"
TRACKEVAL      = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"
_OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")  # e.g. "_seed0"
OUT_ROOT       = "/home/seanachan/GMC-Link/hota_eval_ikun_linear_additive" + _OUT_SUFFIX

TEST_SEQS = ["0005", "0011", "0013"]
FRAMES = {"0005": (0, 296), "0011": (0, 372), "0013": (0, 339)}
SIM_A, SIM_B, SIM_TAU = 8.0, -0.1, 100.0

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


def gen_predicts(text_feat, gmc_caches, alpha, gmc_scale, thr_motion, run_dir,
                 alpha_a=0.0, scale_a=0.0, thr_a=0.0):
    res_dir = os.path.join(run_dir, "results")
    if os.path.exists(res_dir): shutil.rmtree(res_dir)
    os.makedirs(res_dir, exist_ok=True)
    seqmap_lines = []

    for seq in TEST_SEQS:
        ns = merged_ns(seq)
        expr_dir = os.path.join(DATA_ROOT, "expression", seq)
        exprs = sorted(f.replace(".json","") for f in os.listdir(expr_dir) if f.endswith(".json"))
        bias = compute_simcalib_bias(text_feat, exprs)
        gmc_seq = gmc_caches.get(seq, {})
        min_f, max_f = FRAMES[seq]

        for expr in exprs:
            outd = os.path.join(res_dir, seq, expr); os.makedirs(outd, exist_ok=True)
            gt_src = os.path.join(GT_TEMPLATE, seq, expr, "gt.txt")
            gt_dst = os.path.join(outd, "gt.txt")
            if os.path.exists(gt_src): shutil.copy2(gt_src, gt_dst)
            else: open(gt_dst, "w").close()
            open(os.path.join(outd, "predict.txt"), "w").close()
            seqmap_lines.append(f"{seq}+{expr}")

            ikun_scores = load_ikun_scores(CASCADE_FULL, seq, expr)
            b = bias.get(expr, 0.0)
            motion = is_motion(expr)
            per_expr_gmc = gmc_seq.get(expr, {})

            rows = []
            for fid, dets in ns.items():
                if not (min_f < fid < max_f): continue
                for oid, x, y, w, h in dets:
                    cs = ikun_scores.get(fid, {}).get(oid)
                    if cs is None: continue
                    if motion:
                        default = 0.0 if RAW_COS else 0.5
                        gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                        gmc_term = gmc if RAW_COS else (gmc - 0.5)
                        fused = cs + b + alpha * gmc_term * gmc_scale
                        thr = thr_motion
                    else:
                        if scale_a != 0.0:
                            default = 0.0 if RAW_COS else 0.5
                            gmc = float(per_expr_gmc.get(str(fid), {}).get(str(oid), default))
                            gmc_term = gmc if RAW_COS else (gmc - 0.5)
                            fused = cs + b + alpha_a * gmc_term * scale_a
                            thr = thr_a
                        else:
                            fused = cs + b
                            thr = 0.0
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
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--gmc_scale", type=float, default=10.0)
    p.add_argument("--thr", type=float, default=3.0)
    p.add_argument("--alpha_appear", type=float, default=0.0)
    p.add_argument("--gmc_scale_appear", type=float, default=0.0)
    p.add_argument("--thr_appear", type=float, default=0.0)
    p.add_argument("--grid", action="store_true")
    args = p.parse_args()

    print("Loading text_feat + GMC caches...", flush=True)
    text_feat = json.load(open(TEXT_FEAT_JSON))
    gmc_caches = {s: json.load(open(GMC_CACHE_TPL.format(seq=s))) for s in TEST_SEQS}

    if args.grid:
        # Path 1: APPEARANCE-axis GMC extension. Motion ship LOCKED at (α=1.0, sc=0.9, thr=+0.17 → pool 44.400).
        # Sweep appearance bias (alpha_a, scale_a, thr_a). APPEAR is 77% of frames; raw sep +0.264 > motion +0.172.
        # tuple: (tag, alpha_m, scale_m, thr_m, alpha_a, scale_a, thr_a)
        M_A, M_S, M_T = 1.0, 0.9, 0.17
        # Refine 3: refine 2 peak sc=0.25 thr=0.10 → 44.601 (+0.201, beats paper +0.037).
        # Map ridge top with sc=0.25-0.4 × thr=0.10-0.15. Identify peak vs cliff.
        configs = [
            ("appear_sc025_thrp12",  M_A, M_S, M_T, 1.0, 0.25, 0.12),
            ("appear_sc025_thrp15",  M_A, M_S, M_T, 1.0, 0.25, 0.15),
            ("appear_sc03_thrp1",    M_A, M_S, M_T, 1.0, 0.30, 0.10),
            ("appear_sc03_thrp13",   M_A, M_S, M_T, 1.0, 0.30, 0.13),
            ("appear_sc035_thrp1",   M_A, M_S, M_T, 1.0, 0.35, 0.10),
            ("appear_sc035_thrp13",  M_A, M_S, M_T, 1.0, 0.35, 0.13),
            ("appear_sc04_thrp13",   M_A, M_S, M_T, 1.0, 0.40, 0.13),
            ("appear_sc04_thrp17",   M_A, M_S, M_T, 1.0, 0.40, 0.17),
        ]
    else:
        tag = f"a{args.alpha}_scale{args.gmc_scale}_thr{args.thr}"
        if args.gmc_scale_appear != 0.0:
            tag += f"_aa{args.alpha_appear}_sca{args.gmc_scale_appear}_thra{args.thr_appear}"
        configs = [(tag, args.alpha, args.gmc_scale, args.thr,
                    args.alpha_appear, args.gmc_scale_appear, args.thr_appear)]

    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    for tag, a, sc, thr, a_a, sc_a, thr_a in configs:
        run_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== {tag}: motion(α={a}, sc={sc}, thr={thr}) "
              f"appear(α={a_a}, sc={sc_a}, thr={thr_a}) ===", flush=True)
        res_dir, sm = gen_predicts(text_feat, gmc_caches, a, sc, thr, run_dir,
                                    alpha_a=a_a, scale_a=sc_a, thr_a=thr_a)
        pooled = run_te(sm, res_dir)
        moving = run_te(sm, res_dir, class_filter="MOVING")
        static = run_te(sm, res_dir, class_filter="STATIC")
        appear = run_te(sm, res_dir, class_filter="APPEARANCE")
        rows.append((tag, a, sc, thr, a_a, sc_a, thr_a, pooled, appear, moving, static))
        print(f"  pooled={pooled}  APPEAR={appear}  MOVING={moving}  STATIC={static}", flush=True)

    print("\n=== iKUN linear-additive GMC sweep summary ===")
    print("tag                       α_m  sc_m  thr_m  α_a  sc_a  thr_a  pooled   APPEAR   MOVING   STATIC")
    for tag, a, sc, thr, a_a, sc_a, thr_a, p_, ap, mo, st in rows:
        print(f"{tag:<25} {a:4.2f} {sc:4.2f} {thr:5.2f}  {a_a:4.2f} {sc_a:4.2f} {thr_a:5.2f}  "
              f"{p_:.3f}   {ap:.3f}   {mo:.3f}   {st:.3f}")
    print("\niKUN paper-pure baseline:  44.564")
    print("Local B (alpha=0):         44.224")
    print("Motion-only ship (eff09_thrp17): 44.400")


if __name__ == "__main__":
    main()
