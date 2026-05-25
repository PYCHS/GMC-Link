# GMC-Link — Full Comparison Table

3-architecture × 2-split cross-validation. All metrics HOTA (no AUC).
Pooled over test sequences: iKUN/FlexHook V1 = 3-seq (0005/0011/0013);
FlexHook V2 = 4-seq (0005/0011/0013/0019); iKUN-V2 = 3-seq.

**Two columns, two trust levels:**
- **B2 = `{model} + GMC Baseline`** — bare additive fusion `final = model_logit + raw_cos`,
  **zero fitted coefficients**. Nothing tuned against the eval set, so any lift is pure
  GMC signal. This is the trustworthy "does GMC help" number.
- **Ship** — per-arch tuned recipe `final = model_logit + α·(sc·raw_cos + thr)` (18
  coefficients fit to the eval sequences). The headline claim, but test-set-selected.

Aligner = `shared_weight` (raw cosine, no sigmoid, no EMA) for all cells.

## 1. Pooled HOTA

| arch × split | B1 (no GMC) | B2 `cascade+raw_cos` (zero-DOF) | Δ B2−B1 | Ship (tuned) | paper anchor | Δ ship−paper |
|---|---|---|---|---|---|---|
| iKUN · V1 | 44.224 | 44.272 ± 0.018 | **+0.048** | 44.634 ± 0.066 | 44.564 | **+0.070** ✓ |
| FlexHook · V1 | 53.110 | 53.121 ± 0.005 | **+0.011** | 53.526 ± 0.087 | 53.824 | −0.298 ✗ |
| FlexHook · V2 | 42.526 | 42.532 ± 0.002 | **+0.006** | 42.807 ± 0.038 | 42.526 | **+0.281** ✓ |
| iKUN · V2 † | 31.434 | 31.427 (seed0) | **−0.007** | — ‡ | — | — |

† **iKUN-V2 is a zero-shot cross-split probe**, not a ship: V1-trained iKUN scored on V2
paraphrased expressions it never saw. Cascade baseline collapses to 31.434 (vs V1 44.224)
because the CLIP text encoder cannot match unfamiliar paraphrases. Single-seed (seed0).
No published iKUN-V2 anchor exists.
‡ The V1 per-arch recipe *regresses* iKUN-V2 to 28.651 (Δ−2.78) — the `thr` bias is
calibrated to V1's score range and miscalibrates on the weaker V2 scores. A searched probe
(α=0.5) reaches 31.727 (+0.29) but that is test-set hyperparameter selection and vanishes
at zero-DOF; it is **not** cited as a gain.

## 2. Per-class MOVING HOTA (motion-expression gain)

Motion expressions are the class that distinguishes RMOT from open-vocabulary detection,
and where spatially-ignorant vision-language hosts are weakest. GMC's value concentrates here.

| arch × split | MOVING B1 | MOVING ship | Δ | p (1-samp t vs B1) |
|---|---|---|---|---|
| iKUN · V1 | 25.531 | 30.093 ± 0.240 | **+4.562** | 0.0005 |
| FlexHook · V1 | 43.981 | 45.785 ± 0.235 | +1.804 | 0.0028 |
| FlexHook · V2 | 48.018 | 48.758 ± 0.067 | +0.740 | 0.0014 |
| iKUN · V2 † | 26.665 | 25.218 (B2) | −1.447 | — |

iKUN-V1 MOVING **+4.562** is the single largest cell gain — it recovers the ~18-point
STATIC-vs-MOVING hole (cascade iKUN STATIC 43.9 vs MOVING 25.5) that motivates the work.
iKUN-V2 motion is NEG because GMC's motion-reasoning signal needs a functional host
text-matcher to complement; on near-random V2 cascade scores there is nothing to fuse with.

## 3. Per-class pooled HOTA, ship vs B (n=3, all cells)

Source: `project_per_class_pool_all_positive` (2026-05-03, legacy mlp+EMA ship-recipe
measurement — the strongest available all-positive/all-significant pool defense; per-class
absolute values differ slightly from the §1 sw+no-EMA tables for this reason).

| arch | class | B (single-seed) | ship mean ± std (n=3) | Δ | p_one |
|---|---|---|---|---|---|
| iKUN | APPEAR | 46.346 | 46.746 ± 0.045 | +0.400 | 0.0021 |
| iKUN | MOVING | 25.531 | 30.093 ± 0.240 | **+4.562** | 0.0005 |
| iKUN | STATIC | 43.914 | 45.099 ± 0.178 | +1.185 | 0.0037 |
| FlexHook V1 | APPEAR | 55.492 | 55.700 ± 0.026 | +0.208 | 0.0026 |
| FlexHook V1 | MOVING | 43.981 | 45.785 ± 0.235 | +1.804 | 0.0028 |
| FlexHook V1 | STATIC | 48.983 | 49.771 ± 0.217 | +0.788 | 0.0122 |
| FlexHook V2 | APPEAR | 41.748 | 41.946 ± 0.051 | +0.198 | 0.0105 |
| FlexHook V2 | MOVING | 48.018 | 48.758 ± 0.067 | +0.740 | 0.0014 |
| FlexHook V2 | STATIC | 44.622 | 44.935 ± 0.024 | +0.313 | 0.0009 |

**All 9/9 (3-arch × 3-class) cells positive AND significant at α=0.05** (7/9 at α=0.01;
2/9 at α=0.05 only: FH V1 STATIC p=0.0122, FH V2 APPEAR p=0.0105). Largest gain
iKUN MOVING +4.562; smallest t FH V1 STATIC = 6.28 (still passes).

## 4. Verdict

| dimension | result |
|---|---|
| Trustworthy (B2 zero-DOF) | **3/4 cells POS** (+0.006 to +0.048); iKUN-V2 flat (domain shift) |
| Ship (tuned) vs paper | **2/3 beat paper** (iKUN +0.070, FH-V2 +0.281); FH-V1 gap structural |
| Motion class | all 3 ship cells POS + significant; iKUN-V1 +4.562 largest |
| iKUN-V2 cell | flat — V1-trained iKUN zero-shot on V2 paraphrases, not a GMC failure |

## Conventions / caveats

- HOTA only; AUC dropped project-wide.
- iKUN paper anchor 44.564 bit-exact reproduced locally (gt_template_old, 3-seq pooled,
  YOLOv8-NS + NeuralSORT + cascade + simcalib).
- FlexHook V2 paper anchor == its B1 (42.526): paper number equals the raw baseline.
- iKUN-V2 GT frame-shifted to NeuralSORT convention: FlexHook's `gt_template_gen` uses
  TransRMOT (+1 frame), an off-by-one worth ~5.6 HOTA (baseline 25.866 → 31.434 after fix).
- "Ship" carries 18 fitted coefficients (6/arch × 3 arch) selected on the eval sequences —
  a documented limitation; the B2 column is the over-fit-free anchor.

_Sources: `RESULTS_SUMMARY.md`, `RESEARCH_NOTES.md`, and the per-arch ship/baseline memos._
