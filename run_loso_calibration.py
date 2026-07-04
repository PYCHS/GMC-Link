"""LOSO (leave-one-sequence-out) calibration of iKUN fusion coefficients.

Paper E1: answers the "six coefficients are calibrated on the evaluation
data" review attack. Per fold, one V1 test sequence is held out; the
effective coefficients (sc_m, thr_m, sc_a, thr_a; alpha fixed at 1 since
alpha and sc multiply) are fit by grid search on the POOLED HOTA of the
two in-fold sequences only; the held-out sequence is then scored under the
frozen out-of-fold coefficients. A composite prediction set (each sequence
under its own out-of-fold coefficients) yields one LOSO pooled HOTA per
seed, directly comparable to the in-sample ship number 44.634 +- 0.066.

Fit protocol (per fold, per seed), two-round coordinate descent:
  round 1: sweep motion axis with appearance fusion OFF (axes decouple:
           appearance coefficients cannot change motion-expression rows);
  round 2: sweep appearance axis at the fold-best motion setting.
No ship value enters the fit: grids are symmetric around the axis scale,
and the in-fold objective never sees the held-out sequence.

Usage:
    python run_loso_calibration.py --seeds 0 1 2
    python run_loso_calibration.py --seeds 0 --motion-only   # quick probe
"""
import argparse, itertools, json, os, shutil, sys, time

os.environ["GMC_RAW_COS"] = "1"  # ship caches are raw-cosine; must precede import

sys.path.insert(0, "/home/seanachan/GMC-Link")
import run_ikun_linear_additive as ila

TEST_SEQS = ila.TEST_SEQS  # ["0005", "0011", "0013"]
CACHE_TPL = "/home/seanachan/GMC-Link/gmc_link/gmc_scores_v1_{seq}_sharedweight_seed{seed}_rawcos_cache.json"
OUT_ROOT = "/home/seanachan/GMC-Link/hota_eval_loso_calibration"

# Round-1 motion grid (appearance off). alpha=1; sc is the effective scale.
MOTION_GRID = [(sc, thr) for sc in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
               for thr in (0.10, 0.17, 0.25)]
# Round-2 appearance grid at fold-best motion.
APPEAR_GRID = [(sc, thr) for sc in (0.15, 0.22, 0.30, 0.40)
               for thr in (0.05, 0.10, 0.15)]

_bias_memo = {}
_orig_bias = ila.compute_simcalib_bias


def _memo_bias(text_feat, exprs):
    key = tuple(exprs)
    if key not in _bias_memo:
        _bias_memo[key] = _orig_bias(text_feat, exprs)
    return _bias_memo[key]


ila.compute_simcalib_bias = _memo_bias


def seq_seqmap(master_sm, run_dir, seqs, tag, moving_only=False):
    lines = [l for l in open(master_sm).read().splitlines()
             if l and l.split("+", 1)[0] in seqs
             and (not moving_only or ila.classify(l.split("+", 1)[1]) == "MOVING")]
    if not lines:
        return None
    sm = os.path.join(run_dir, f"seqmap_{tag}.txt")
    open(sm, "w").write("\n".join(lines) + "\n")
    return sm


def eval_subset(master_sm, res_dir, run_dir, seqs, tag, moving_only=False):
    sm = seq_seqmap(master_sm, run_dir, seqs, tag, moving_only)
    if sm is None:
        return None
    return ila.run_te(sm, res_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--motion-only", action="store_true",
                    help="round 1 only (quick probe)")
    args = ap.parse_args()

    text_feat = json.load(open(ila.TEXT_FEAT_JSON))
    results = {}

    for seed in args.seeds:
        t_seed = time.time()
        gmc_caches = {s: json.load(open(CACHE_TPL.format(seq=s, seed=seed)))
                      for s in TEST_SEQS}
        run_dir = os.path.join(OUT_ROOT, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        folds = {held: [s for s in TEST_SEQS if s != held] for held in TEST_SEQS}

        # ---- round 1: motion sweep, appearance off, scored on every pair ----
        pair_scores = {}   # (sc, thr) -> {held: pooled HOTA of in-fold pair}
        for sc, thr in MOTION_GRID:
            res_dir, sm = ila.gen_predicts(text_feat, gmc_caches, 1.0, sc, thr,
                                           run_dir)
            row = {}
            for held, pair in folds.items():
                row[held] = eval_subset(sm, res_dir, run_dir, pair,
                                        f"pair_no{held}")
            pair_scores[(sc, thr)] = row
            print(f"[seed{seed}] motion sc={sc} thr={thr} "
                  + " ".join(f"fit(no {h})={v}" for h, v in row.items()),
                  flush=True)

        best_motion = {held: max(MOTION_GRID,
                                 key=lambda c: pair_scores[c][held] or -1)
                       for held in TEST_SEQS}
        print(f"[seed{seed}] fold-best motion: {best_motion}", flush=True)
        if args.motion_only:
            results[seed] = {"best_motion": {h: list(c) for h, c in best_motion.items()}}
            continue

        # ---- round 2: appearance sweep at each fold's best motion ----
        # Folds sharing a motion setting share generations.
        best_appear, held_out = {}, {}
        comp_dir = os.path.join(run_dir, "composite")
        if os.path.exists(comp_dir):
            shutil.rmtree(comp_dir)
        os.makedirs(comp_dir)
        for motion_c in set(best_motion.values()):
            holds = [h for h, c in best_motion.items() if c == motion_c]
            sc_m, thr_m = motion_c
            app_scores = {h: {} for h in holds}
            for sc_a, thr_a in APPEAR_GRID:
                res_dir, sm = ila.gen_predicts(
                    text_feat, gmc_caches, 1.0, sc_m, thr_m, run_dir,
                    alpha_a=1.0, scale_a=sc_a, thr_a=thr_a)
                for held in holds:
                    v = eval_subset(sm, res_dir, run_dir, folds[held],
                                    f"pair_no{held}")
                    app_scores[held][(sc_a, thr_a)] = v
                print(f"[seed{seed}] appear sc={sc_a} thr={thr_a} @motion{motion_c} "
                      + " ".join(f"fit(no {h})={app_scores[h][(sc_a, thr_a)]}"
                                 for h in holds), flush=True)
            for held in holds:
                best_appear[held] = max(APPEAR_GRID,
                                        key=lambda c: app_scores[held][c] or -1)

        # ---- held-out evaluation + composite assembly ----
        for held in TEST_SEQS:
            sc_m, thr_m = best_motion[held]
            sc_a, thr_a = best_appear[held]
            res_dir, sm = ila.gen_predicts(
                text_feat, gmc_caches, 1.0, sc_m, thr_m, run_dir,
                alpha_a=1.0, scale_a=sc_a, thr_a=thr_a)
            pooled = eval_subset(sm, res_dir, run_dir, [held], f"held_{held}")
            moving = eval_subset(sm, res_dir, run_dir, [held], f"heldmov_{held}",
                                 moving_only=True)
            held_out[held] = {"motion": [sc_m, thr_m], "appear": [sc_a, thr_a],
                              "pooled": pooled, "moving": moving}
            shutil.copytree(os.path.join(res_dir, held),
                            os.path.join(comp_dir, held))
            print(f"[seed{seed}] HELD-OUT {held}: motion(sc={sc_m},thr={thr_m}) "
                  f"appear(sc={sc_a},thr={thr_a}) pooled={pooled} moving={moving}",
                  flush=True)

        # composite: every seq under its own out-of-fold coefficients
        master = seq_seqmap(sm, run_dir, TEST_SEQS, "composite_all")
        comp_pooled = ila.run_te(master, comp_dir)
        mov_sm = seq_seqmap(sm, run_dir, TEST_SEQS, "composite_mov",
                            moving_only=True)
        comp_moving = ila.run_te(mov_sm, comp_dir) if mov_sm else None
        results[seed] = {"held_out": held_out,
                         "loso_pooled": comp_pooled,
                         "loso_moving": comp_moving}
        print(f"[seed{seed}] LOSO pooled={comp_pooled} moving={comp_moving} "
              f"({time.time()-t_seed:.0f}s)", flush=True)

    out = os.path.join(OUT_ROOT, "loso_results.json")
    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    pooled = [r["loso_pooled"] for r in results.values() if "loso_pooled" in r]
    if pooled:
        import statistics as st
        m = st.mean(pooled)
        s = st.stdev(pooled) if len(pooled) > 1 else 0.0
        print(f"LOSO pooled HOTA n={len(pooled)}: {m:.3f} +- {s:.3f} "
              f"(in-sample ship 44.634 +- 0.066)")


if __name__ == "__main__":
    main()
