"""Per-class POOL HOTA × 3 seeds × 3 archs + 3 single-B baselines.

Computes pool HOTA on per-class seqmaps (APPEARANCE / MOVING / STATIC) for:
  - Ship: 3 archs × 3 seeds (multi-seed)
  - B (no GMC): 3 archs × 1 (no aligner dep)

Then reports per-class Δ = ship − B with mean ± std and 1-sample t-test vs B.
"""
import os, subprocess, sys
import numpy as np
from scipy import stats

TRACKEVAL = "/home/seanachan/TempRMOT/TrackEval/scripts/run_mot_challenge.py"

ARCHS = {
    "iKUN": {
        "B":    "/home/seanachan/GMC-Link/hota_eval_ikun_linear_additive/baseline_a0",
        "ship": [
            f"/home/seanachan/GMC-Link/hota_eval_ikun_linear_additive_seed{i}/"
            "a1.0_scale0.9_thr0.17_aa1.0_sca0.3_thra0.1"
            for i in range(3)
        ],
    },
    "V1": {
        "B":    "/home/seanachan/GMC-Link/hota_eval_flexhook_phase5_gmc/baseline_a0_thr0",
        "ship": [
            f"/home/seanachan/GMC-Link/hota_eval_flexhook_phase5_gmc_seed{i}/"
            "a0.65_scale10.0_thr3.0_aa1.0_sca3.5_thra0.9"
            for i in range(3)
        ],
    },
    "V2": {
        "B":    "/home/seanachan/GMC-Link/hota_eval_flexhook_v2_raw_gmc/baseline_a0_thr0",
        "ship": [
            f"/home/seanachan/GMC-Link/hota_eval_flexhook_v2_raw_gmc_seed{i}/"
            "a0.4_scale10.0_thr1.3_aa1.0_sca3.5_thra1.2"
            for i in range(3)
        ],
    },
}

CLASSES = ["APPEARANCE", "MOVING", "STATIC"]


def run_te(run_dir, cls):
    res = os.path.abspath(os.path.join(run_dir, "results"))
    sm = os.path.abspath(os.path.join(run_dir, f"seqmap_{cls}.txt"))
    if not os.path.exists(sm) or os.path.getsize(sm) == 0:
        return None
    sp = os.path.join(res, "pedestrian_summary.txt")
    if os.path.exists(sp): os.remove(sp)
    cmd = [sys.executable, TRACKEVAL, "--METRICS", "HOTA",
           "--SEQMAP_FILE", sm, "--SKIP_SPLIT_FOL", "True",
           "--GT_FOLDER", res, "--TRACKERS_FOLDER", res,
           "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
           "--TRACKERS_TO_EVAL", res,
           "--USE_PARALLEL", "False", "--PLOT_CURVES", "False",
           "--PRINT_CONFIG", "False"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=os.path.dirname(TRACKEVAL))
    if not os.path.exists(sp):
        sys.stderr.write(f"FAIL {run_dir} {cls} rc={proc.returncode}\n{proc.stderr[-800:]}\n")
        return None
    with open(sp) as fh:
        return float(fh.read().splitlines()[1].split()[0])


def main():
    results = {}
    for arch, cfg in ARCHS.items():
        print(f"\n=== {arch} ===", flush=True)
        b_path = cfg["B"]
        b_vals = {}
        for cls in CLASSES:
            v = run_te(b_path, cls)
            b_vals[cls] = v
            print(f"  B  {cls:<11} = {v}", flush=True)
        ship_vals = {cls: [] for cls in CLASSES}
        for i, sp in enumerate(cfg["ship"]):
            for cls in CLASSES:
                v = run_te(sp, cls)
                ship_vals[cls].append(v)
                print(f"  s{i} {cls:<11} = {v}", flush=True)
        results[arch] = {"B": b_vals, "ship": ship_vals}

    print("\n\n=== Per-class POOL Δ ship − B (multi-seed) ===")
    print(f"{'arch':<6} {'class':<11} {'B':>8} {'s0':>8} {'s1':>8} {'s2':>8} "
          f"{'mean':>8} {'std':>7} {'Δ':>7} {'t':>6} {'p_one':>7}")
    rows = []
    for arch, d in results.items():
        for cls in CLASSES:
            B = d["B"][cls]
            ss = d["ship"][cls]
            if B is None or any(s is None for s in ss):
                print(f"{arch:<6} {cls:<11} SKIP")
                continue
            arr = np.array(ss, dtype=float)
            m, sd = arr.mean(), arr.std(ddof=1)
            delta = m - B
            # one-sample t (df=2): t = (m − B) / (sd/√n)
            t = delta / (sd / np.sqrt(len(arr))) if sd > 0 else float("inf")
            # one-sided p (m > B)
            p_one = 1 - stats.t.cdf(t, df=len(arr)-1) if sd > 0 else 0.0
            sig = "✓✓" if p_one < 0.01 else ("✓" if p_one < 0.05 else "")
            print(f"{arch:<6} {cls:<11} {B:>8.3f} {ss[0]:>8.3f} {ss[1]:>8.3f} {ss[2]:>8.3f} "
                  f"{m:>8.3f} {sd:>7.3f} {delta:>+7.3f} {t:>6.2f} {p_one:>7.4f} {sig}")
            rows.append((arch, cls, B, m, sd, delta, t, p_one))

    n_pos = sum(1 for r in rows if r[5] > 0)
    n_sig = sum(1 for r in rows if r[7] < 0.05)
    print(f"\n{n_pos}/{len(rows)} cells POS  |  {n_sig}/{len(rows)} cells stat-sig at α=0.05")


if __name__ == "__main__":
    main()
