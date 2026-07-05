"""LOSO calibration of FlexHook fusion coefficients (V1 3-fold / V2 4-fold).

FlexHook counterpart of run_loso_calibration.py (see that docstring for the
protocol). Fusion form: fused = margin + alpha*gmc*scale, keep iff fused > thr
(GMC_RAW_COS=1 caches). alpha and scale multiply, so alpha is fixed at 1 and
the grid sweeps the effective scale; thr is a separate admission gate.
Round 1 sweeps the motion axis with the appearance axis off; round 2 sweeps
appearance at each fold's best motion. STATIC keeps ship behaviour (inherits
the motion bias; no static axis). Composite = each held-out sequence under
its own out-of-fold coefficients -> one LOSO pooled HOTA per seed.

Usage:
    python run_loso_calibration_flexhook.py --host v1 --seeds 0 1 2
    python run_loso_calibration_flexhook.py --host v2 --seeds 0 1 2
"""
import argparse, json, os, shutil, sys, time

os.environ["GMC_RAW_COS"] = "1"  # must precede module import

sys.path.insert(0, "/home/seanachan/GMC-Link")

HOSTS = {
    "v1": {
        "module": "run_flexhook_phase5_gmc_sweep",
        "cache_tpl": "/home/seanachan/GMC-Link/gmc_link/gmc_scores_flexhook_v1_{seq}_sharedweight_seed{seed}_rawcos_cache.json",
        "out_root": "/home/seanachan/GMC-Link/hota_eval_loso_flexhook_v1",
        # ship: motion α=0.65 sc=10 (eff 6.5) thr=+3; appear α=1 sc=3.5 thr=+0.9
        "motion_grid": [(sc, thr) for sc in (3.0, 5.0, 6.5, 8.0, 10.0)
                        for thr in (1.5, 3.0, 4.5)],
        "appear_grid": [(sc, thr) for sc in (1.75, 3.5, 5.25)
                        for thr in (0.45, 0.9, 1.35)],
        "ship_ref": "in-sample ship 53.526 +- 0.087",
    },
    "v2": {
        "module": "run_flexhook_v2_raw_sweep",
        "cache_tpl": "/home/seanachan/GMC-Link/gmc_link/gmc_scores_flexhook_v2_raw_{seq}_sharedweight_seed{seed}_rawcos_cache.json",
        "out_root": "/home/seanachan/GMC-Link/hota_eval_loso_flexhook_v2",
        # ship: motion α=0.4 sc=10 (eff 4.0) thr=+1.3; appear α=1 sc=3.5 thr=+1.2
        "motion_grid": [(sc, thr) for sc in (2.0, 3.0, 4.0, 5.0, 6.5)
                        for thr in (0.65, 1.3, 2.0)],
        "appear_grid": [(sc, thr) for sc in (1.75, 3.5, 5.25)
                        for thr in (0.6, 1.2, 1.8)],
        "ship_ref": "in-sample ship 42.807 +- 0.038",
    },
}


def seq_seqmap(host, master_sm, run_dir, seqs, tag, moving_only=False):
    lines = [l for l in open(master_sm).read().splitlines()
             if l and l.split("+", 1)[0] in seqs
             and (not moving_only or host.classify(l.split("+", 1)[1]) == "MOVING")]
    if not lines:
        return None
    sm = os.path.join(run_dir, f"seqmap_{tag}.txt")
    open(sm, "w").write("\n".join(lines) + "\n")
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", choices=("v1", "v2"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    cfg = HOSTS[args.host]

    import importlib
    host = importlib.import_module(cfg["module"])

    print(f"Loading {cfg['module']} inputs (result_0.json ~80MB)...", flush=True)
    cls_dict = json.load(open(host.RESULT_JSON))
    tracks_by_seq = {s: host.load_tracks(s) for s in host.TEST_SEQS}

    def eval_subset(master_sm, res_dir, run_dir, seqs, tag, moving_only=False):
        sm = seq_seqmap(host, master_sm, run_dir, seqs, tag, moving_only)
        return host.run_te(sm, res_dir) if sm else None

    results = {}
    for seed in args.seeds:
        t_seed = time.time()
        gmc_caches = {s: json.load(open(cfg["cache_tpl"].format(seq=s, seed=seed)))
                      for s in host.TEST_SEQS}
        run_dir = os.path.join(cfg["out_root"], f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        folds = {h: [s for s in host.TEST_SEQS if s != h] for h in host.TEST_SEQS}

        def gen(sc_m, thr_m, sc_a=0.0, thr_a=0.0):
            return host.gen_predicts(cls_dict, tracks_by_seq, gmc_caches,
                                     1.0, sc_m, thr_m, run_dir,
                                     alpha_a=1.0 if sc_a else 0.0,
                                     scale_a=sc_a, thr_a=thr_a)

        # round 1: motion sweep, appearance off
        pair_scores = {}
        for sc, thr in cfg["motion_grid"]:
            res_dir, sm = gen(sc, thr)
            row = {h: eval_subset(sm, res_dir, run_dir, pair, f"pair_no{h}")
                   for h, pair in folds.items()}
            pair_scores[(sc, thr)] = row
            print(f"[{args.host} seed{seed}] motion sc={sc} thr={thr} "
                  + " ".join(f"fit(no {h})={v}" for h, v in row.items()), flush=True)
        best_motion = {h: max(cfg["motion_grid"],
                              key=lambda c: pair_scores[c][h] or -1)
                       for h in host.TEST_SEQS}
        print(f"[{args.host} seed{seed}] fold-best motion: {best_motion}", flush=True)

        # round 2: appearance sweep at fold-best motion (shared gens per motion setting)
        best_appear, held_out = {}, {}
        comp_dir = os.path.join(run_dir, "composite")
        if os.path.exists(comp_dir):
            shutil.rmtree(comp_dir)
        os.makedirs(comp_dir)
        for motion_c in set(best_motion.values()):
            holds = [h for h, c in best_motion.items() if c == motion_c]
            app_scores = {h: {} for h in holds}
            for sc_a, thr_a in cfg["appear_grid"]:
                res_dir, sm = gen(*motion_c, sc_a=sc_a, thr_a=thr_a)
                for h in holds:
                    app_scores[h][(sc_a, thr_a)] = eval_subset(
                        sm, res_dir, run_dir, folds[h], f"pair_no{h}")
                print(f"[{args.host} seed{seed}] appear sc={sc_a} thr={thr_a} "
                      f"@motion{motion_c} "
                      + " ".join(f"fit(no {h})={app_scores[h][(sc_a, thr_a)]}"
                                 for h in holds), flush=True)
            for h in holds:
                best_appear[h] = max(cfg["appear_grid"],
                                     key=lambda c: app_scores[h][c] or -1)

        # held-out eval + composite assembly
        for h in host.TEST_SEQS:
            sc_m, thr_m = best_motion[h]
            sc_a, thr_a = best_appear[h]
            res_dir, sm = gen(sc_m, thr_m, sc_a=sc_a, thr_a=thr_a)
            pooled = eval_subset(sm, res_dir, run_dir, [h], f"held_{h}")
            moving = eval_subset(sm, res_dir, run_dir, [h], f"heldmov_{h}",
                                 moving_only=True)
            held_out[h] = {"motion": [sc_m, thr_m], "appear": [sc_a, thr_a],
                           "pooled": pooled, "moving": moving}
            shutil.copytree(os.path.join(res_dir, h), os.path.join(comp_dir, h))
            print(f"[{args.host} seed{seed}] HELD-OUT {h}: "
                  f"motion(sc={sc_m},thr={thr_m}) appear(sc={sc_a},thr={thr_a}) "
                  f"pooled={pooled} moving={moving}", flush=True)

        master = seq_seqmap(host, sm, run_dir, host.TEST_SEQS, "composite_all")
        comp_pooled = host.run_te(master, comp_dir)
        mov_sm = seq_seqmap(host, sm, run_dir, host.TEST_SEQS, "composite_mov",
                            moving_only=True)
        comp_moving = host.run_te(mov_sm, comp_dir) if mov_sm else None
        results[seed] = {"held_out": held_out, "loso_pooled": comp_pooled,
                         "loso_moving": comp_moving}
        print(f"[{args.host} seed{seed}] LOSO pooled={comp_pooled} "
              f"moving={comp_moving} ({time.time()-t_seed:.0f}s)", flush=True)

    os.makedirs(cfg["out_root"], exist_ok=True)
    out = os.path.join(cfg["out_root"], "loso_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWrote {out}")
    pooled = [r["loso_pooled"] for r in results.values()
              if r.get("loso_pooled") is not None]
    if pooled:
        import statistics as st
        m = st.mean(pooled)
        s = st.stdev(pooled) if len(pooled) > 1 else 0.0
        print(f"LOSO pooled HOTA n={len(pooled)}: {m:.3f} +- {s:.3f} "
              f"({cfg['ship_ref']})")


if __name__ == "__main__":
    main()
