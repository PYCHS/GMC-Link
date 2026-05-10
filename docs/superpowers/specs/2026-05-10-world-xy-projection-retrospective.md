# World-XY Projection (F-variant) Retrospective — 2026-05-10

## TL;DR

World-XY F-variant (image-plane dx,dy → metric world dX,dY via inverse pinhole `dX = dx × Z / f_x`) **NEUTRAL** vs 17D depth-aug image-domain at all 3 archs. Hypothesis "metric units help vs image units" **falsified**. Aligner absorbs the linear unit scale through learned VELOCITY_SCALE_WORLD.

**No new ship.** Depth-aug image-domain ship (iKUN) remains the live recipe.

## Spec Reference
- Plan: `docs/superpowers/plans/2026-05-10-world-xy-projection.md`
- Design: `docs/superpowers/specs/2026-05-10-world-xy-projection-design.md`
- Ship gate per §5.5: HOTA pool POS at any arch (NOT AUC)

## Multi-Seed Results (n=3, locked Arm A recipes)

### Pooled HOTA

| arch | depth-aug | world-XY | Δ | t | p_two | p_one_pos |
|------|-----------|----------|---|---|-------|-----------|
| iKUN  | 44.823 (44.876, 44.800, 44.793) | 44.823 (44.909, 44.860, 44.701) | +0.000 | +0.007 | 0.9950 | 0.4975 |
| FH V1 | 53.765 (53.787, 53.809, 53.698) | 53.659 (53.610, 53.625, 53.742) | −0.106 | −1.412 | 0.2936 | 0.8532 |
| FH V2 | 42.833 (42.836, 42.836, 42.828) | 42.834 (42.778, 42.906, 42.817) | +0.000 | +0.009 | 0.9937 | 0.4968 |

### vs Paper

| arch | paper | world-XY | Δ vs paper |
|------|-------|----------|-----------|
| iKUN  | 44.564 | 44.823 | +0.259 |
| FH V1 | 53.110 | 53.659 | +0.549 |
| FH V2 | 42.526 | 42.834 | +0.308 |

All 3 archs beat paper — **same magnitude as depth-aug**, no incremental gain.

### Per-Class HOTA (avg)

| arch | depth APPEAR / MOVING / STATIC | world-XY APPEAR / MOVING / STATIC |
|------|-------------------------------|-----------------------------------|
| iKUN  | 46.79 / 32.07 / 44.86 | 46.74 / 32.14 / 45.30 |
| FH V1 | 55.84 / 45.48 / 49.73 | 55.80 / 44.71 / 49.68 |
| FH V2 | 41.96 / 48.69 / 45.17 | 41.97 / 48.70 / 45.20 |

Per-class also flat — no asymmetric win on motion-class either.

## Stage1 AUC (n=3)

| metric | 13D stage1 | 17D depth-aug image | 17D world-XY | Δ vs depth |
|--------|-----------|--------------------|--------------|-----------|
| micro AUC | 0.7793 | 0.7567 | 0.7558 ± 0.0034 | −0.0009 |
| macro AUC | 0.840  | 0.819  | 0.8253 ± 0.0061 | +0.006 |

AUC per spec KILL gate < 0.760 trigged but DO NOT kill — depth-aug precedent showed AUC NEG → HOTA POS at iKUN. World-XY broke that pattern: HOTA also flat.

## Why Did It Fail?

Three plausible causes (cannot disambiguate from this single experiment):

1. **Aligner absorbs linear scale.** The 13→256 motion projector is linear in the input. Multiplying motion[0:6] by `Z / f_x ≈ 30/721 ≈ 0.042` is a per-dim scale that VELOCITY_SCALE_WORLD can re-absorb during training. Image-domain (`v / 100`) and world-domain (`v × Z / f_x × 2.0`) produce equivalent learned representations.

2. **Z signal too noisy at fixed-K calibration.** KITTI's depth (DAv2 metric) has occlusion gaps + scale uncertainty (~10-15% relative). Image dx,dy is observable noise-free; multiplying by noisy Z could degrade signal more than the metric-unit interpretability buys.

3. **Object scale invariant of unit choice for cosine similarity.** The aligner produces L2-normalized 256D embeddings, then InfoNCE matches on cosine. Cosine is unit-invariant up to per-dim magnitude — and per-dim magnitude is set by the projector, not the input.

The ablation that would disambiguate (1) vs (2,3): freeze projector after image-domain training, fine-tune only on world-XY samples. NOT pursued — Δ=0 already flat ship gate.

## Cross-Arch Sign

7/9 cells fall within ±0.1 HOTA of depth-aug. No sign-flip pattern. Truly equivalent.

## Code Status

All World-XY code stays merged on branch `exp/ego-motion-systematic`:
- `gmc_link/camera_intrinsics.py` (NEW) — KITTI canonical 2011_09_26 P_rect_02
- `gmc_link/manager.py` — `world_xy=True` path swaps slots [0:6] via inverse pinhole
- `gmc_link/dataset.py`, `gmc_link/train.py` — `--world-xy` plumbing
- `diagnostics/diag_gt_cosine_distributions.py` — inline projection at compute_motion_vectors
- 3 cache builders thread `world_xy` from ckpt meta

Code is dormant but functional. Future experiments wanting metric-domain motion can re-enable via `--world-xy` flag without re-implementing.

## Decision

- **No new ship recipe.** Depth-aug iKUN ship (mean 44.823, +0.215 sig vs 13D) remains live.
- **Depth-aug FH V1 / V2** (within-noise +0.048 / +0.034) likewise remain live; world-XY does NOT change those decisions.
- **Hypothesis falsified:** image-plane (dx, dy) is sufficient — metric (dX, dY) provides no incremental signal at the cosine-similarity output level.

## Lessons

- **Linear-projector input transforms = no-ops** at fixed input dim. The aligner can re-learn any per-dim scale via projector weights.
- **HOTA gate matters more than AUC** (depth-aug precedent reinforced) — but world-XY shows NEUTRAL at both gates, so no rescue here.
- **DAv2 metric depth in 13D bbox slots [10:13]** gives the gain (per depth-aug retrospective). Adding metric to dynamic slots [0:6] is redundant.
- 14 levers exhausted at stage1 (now +1 = 15). Ceiling 0.7793 micro AUC remains pipeline-bound.
