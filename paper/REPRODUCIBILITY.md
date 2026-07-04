# REPRODUCIBILITY.md — every paper number → command / config / checkpoint / log

Maps each number reported in `paper/latex/main.tex` (MMAsia '26 draft) to the exact
repo command, env vars, weight checkpoints, score caches, and logs that produced it.
All commands run from repo root. Data paths per `CLAUDE.md` (Refer-KITTI at
`/home/seanachan/data/Dataset/refer-kitti-v2`, GT template `gt_template_old`,
NeuralSORT detections in `NeuralSORT/`).

## Shared components

**Aligner training (ship arch `shared_weight`, per seed N):**
```bash
python -m gmc_link.train --split v1 --stage 1 --architecture shared_weight \
    --seed N --save-path gmc_link_weights_v1train_sharedweight_seedN.pth
```
Checkpoints: `gmc_link_weights_v1train_sharedweight_seed{0..4}.pth`.
Sample training log (seed 3): `paper/repro_logs/train_full_seed3.log`.

**iKUN GMC caches (per seed N, per seq S ∈ {0005,0011,0013}):**
```bash
GMC_WEIGHTS=gmc_link_weights_v1train_sharedweight_seedN.pth \
GMC_SUFFIX=_sharedweight_seedN_rawcos GMC_RAW_COS=1 \
    python run_build_gmc_cache.py S
```
Caches: `gmc_link/gmc_scores_v1_{S}_sharedweight_seed{N}_rawcos_cache.json`.

**iKUN ship eval (pooled + per-class HOTA, per seed N):**
```bash
GMC_SUFFIX=_sharedweight_seedN_rawcos GMC_RAW_COS=1 \
    python run_ikun_linear_additive.py \
        --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
        --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10
```

## Table 1 — pooled HOTA (n=3, seeds 0–2)

| Number | Producer |
|---|---|
| iKUN native 44.564 | Published (iKUN repo README, cascade+sim_calib YOLOv8-NS anchor; local repro 44.224, see memory `project_paper_repro_3seq_pooled`) |
| iKUN +GMC simple 44.272 | iKUN ship eval cmd with `--alpha 1.0 --gmc_scale 1.0 --thr 0.0` (no appearance axis), seeds 0–2 mean |
| iKUN +GMC ship 44.634 ± 0.066 | iKUN ship eval cmd above, seeds 0–2 (2026-05-21 ship record). **Drift note:** 2026-07-04 re-eval of identical seeds/caches gives 44.580 (per-seed 44.561/44.513/44.667, `paper/repro_logs/e2_seeds012.log`); MOVING reproduces exactly. Cause on appearance/static side, unresolved. |
| FH (V1) native 53.824 | Published (FlexHook paper, V1) |
| FH (V1) +GMC simple 53.121 | `run_flexhook_phase5_gmc_sweep.py` with α=1, sc=1, thr=0, seeds 0–2 |
| FH (V1) ship 53.526 ± 0.087 | `GMC_SUFFIX=_sharedweight_seedN_rawcos GMC_RAW_COS=1 python run_flexhook_phase5_gmc_sweep.py --alpha 0.65 --gmc_scale 10.0 --thr 3.0 --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9`; caches via `run_build_gmc_cache_flexhook.py` |
| FH (V2) native 42.526 | Published (FlexHook paper, V2) |
| FH (V2) +GMC simple 42.532 | `run_flexhook_v2_raw_sweep.py` with α=1, sc=1, thr=0, seeds 0–2 |
| FH (V2) ship 42.807 ± 0.038 | `GMC_SUFFIX=_sharedweight_seedN_rawcos GMC_RAW_COS=1 python run_flexhook_v2_raw_sweep.py --alpha 0.4 --gmc_scale 10.0 --thr 1.3 --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2`; caches via `run_build_gmc_cache_flexhook_v2_raw.py` |

## Table 2 — fusion coefficients

Hand-calibrated constants; live in the ship eval commands above (locked recipes,
also recorded in `CLAUDE.md` → "Ship recipes").

## Table 3 — motion deficit (MOVING HOTA)

| Number | Producer |
|---|---|
| iKUN native MOVING 20.25 | iKUN eval with module off (α=0): `python run_ikun_linear_additive.py --alpha 0 --gmc_scale 0 --thr 0`, MOVING readout |
| iKUN gain +8.83 ± 0.83 (n=5) | Full-module MOVING per seed (Table 4 full arm) − 20.25; seeds 0–4; logs `paper/repro_logs/e2_seeds012.log` (seeds 0–2) + `paper/repro_logs/e2_chain.log` (seeds 3–4) |
| FH (V1) native MOVING 43.98 / gain +0.91 ± 0.79 | FH V1 eval cmd, MOVING readout, α=0 vs ship recipe, seeds 0–2 |
| FH (V2) native MOVING 48.02 / gain +0.43 ± 0.17 | FH V2 eval cmd, MOVING readout, α=0 vs ship recipe, seeds 0–2 |

## Table 4 — component ablation (iKUN MOVING HOTA, n=5, seeds 0–4)

Arm construction: train + cache-build + eval all run under the arm's env vars
(hooks in `gmc_link/manager.py` / `gmc_link/dataset.py`, commit `5b39455`).

| Arm | Env at train+cache | Checkpoints | MOVING (n=5) |
|---|---|---|---|
| Full module | (none) | `gmc_link_weights_v1train_sharedweight_seed{0..4}.pth` | 29.08 ± 0.83 |
| −ego | `GMC_RAWVEL=1` | `gmc_w_noego_seed{0..4}.pth` | 27.31 ± 0.66 |
| −multiscale | `GMC_GAPS=5,5,5` | `gmc_w_singlegap_seed{0..4}.pth` | 28.82 ± 0.84 |
| −snr | `GMC_NO_SNR=1` | `gmc_w_nosnr_seed{0..4}.pth` | 29.74 ± 0.59 |

Caches: `gmc_link/gmc_scores_v1_{S}_{noego|singlegap|nosnr}_seed{N}_rawcos_cache.json`.
Eval = iKUN ship eval cmd with the arm's `GMC_SUFFIX`.
Per-seed values + Welch t (ego: t=3.75, df≈7.6, p=0.006):
seeds 3–4 pipeline log `paper/repro_logs/e2_chain.log`; seeds 0–2 re-evals
`paper/repro_logs/e2_seeds012.log`; n=3-era record in memory
`project_ablation_moving_hota_n3_2026_06_24`.

## Sec 4.1 — LOSO calibration check

```bash
python run_loso_calibration.py --seeds 0 1 2
```
Script: `run_loso_calibration.py` (grids + protocol in docstring).
Results JSON: `hota_eval_loso_calibration/loso_results.json` (committed, c715e64).
Full sweep log: `paper/repro_logs/loso.log`.
Numbers: LOSO pooled 44.316 ± 0.092; LOSO MOVING 28.38 ± 1.60 (per-seed
26.613/28.779/29.740); in-sample same-seed reference 29.47 ± 0.88; fold-fit
coefficients printed per fold in the log ("HELD-OUT" lines); ship per-seq
references printed by the inline snippet recorded in the log tail of the session
(seed{0..2} SHIP lines; also reproducible via `run_loso_calibration.py` internals).

## Inline numbers

| Number | Producer |
|---|---|
| Expression counts / class mix (V1 158 exprs, 75.3/7.6/17.1; V2 862, 74.1/11.7/14.2) | Refer-KITTI expression JSONs (`refer-kitti/expression/`), classified by the ship router in `run_ikun_linear_additive.py:60` (`MOTION_KW`/`STATIC_KW`) |
| Nine per-class pooled deltas all positive (n=3) | `run_per_class_multiseed_eval.py`; record: memory `project_per_class_pool_all_positive` |
| Feature-level injection −21.7% F1 | Historical diagnostic, prior pipeline (RESEARCH_NOTES.md Exp — feature-level CLIP injection); not rerun on ship stack |
| Appearance re-ranker stack 45.612 | CLIP-L/14 spatial rerank, commits 2091c84/efc7372/9699e7b; record: memory `project_appearance_rerank_clipL14_2026_06_01` |
| GMC module 57–61 FPS CPU | Profiling over ORB+homography+descriptor+alignment stages (RESEARCH_NOTES.md) |
| VMRMOT 53.00/35.21 | VMRMOT paper (arXiv 2511.17681) |
| std-matching auto-derive NEG | memory `project_variant_b_std_matching_negative_2026_05_21` |
| Learned fusion head NEG (HOTA −3.79) | `gmc_link/fusion_head.py` (legacy); memory `project_flexhook_learned_fusion_negative` |
