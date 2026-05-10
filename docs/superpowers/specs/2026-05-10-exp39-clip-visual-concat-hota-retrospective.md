# Exp 39 CLIP-Visual Concat HOTA Revisit Retrospective — 2026-05-10

## TL;DR

CLIP B/32 DataComp-XL visual concat into 13D motion (Exp 39 clip128 ckpt, AUC=0.7406)
**FALSIFIED at HOTA gate** across all 3 archs. Δ vs depth-aug single-seed: iKUN
−0.139, FH V1 −0.096, FH V2 −0.229. All 3 flat-NEG; none reach +0.2 escalation gate.
iKUN MOVING-class regresses −2.7 HOTA. Reinforces aligner-internal fusion NEG pattern
(Exp 39 AUC, Exp 40, Exp 41).

**No new ship.** Depth-aug iKUN remains live recipe.

## Context

Depth-aug precedent (2026-05-10) proved AUC NEG can survive at HOTA: iKUN +0.215 sig
despite micro AUC −0.0226. User asked: should Exp 39 (killed on AUC=0.7223 / 0.7406)
be revisited at HOTA? This experiment answers: **NO**.

## Setup

- Ckpt: `experiments/exp39_clip128/weights.pth` (clip_proj_dim=128, AUC 0.7406, stronger of two clip widths)
- Patch surface: `gmc_link/manager.py` (lazy CLIP B/32 load, runtime bbox-crop encode), `run_build_gmc_cache_flexhook_v2_raw.py` (parallel clip_feats list through 2-pass split)
- Cache build: 10 caches (3 iKUN + 3 FH V1 + 4 FH V2), ~70min total
- Eval: locked Arm A recipes per arch, 3-seq pool HOTA (mandatory ship gate)

## Results

### Pooled HOTA (single-seed reference)

| arch | depth-aug | exp39 clip128 | Δ vs depth | t-suggested? |
|------|-----------|---------------|------------|--------------|
| iKUN | 44.876 | 44.737 | −0.139 | NEG |
| FH V1 | 53.787 | 53.691 | −0.096 | flat-NEG |
| FH V2 | 42.836 | 42.607 | −0.229 | NEG |

### vs Paper (still POS, but no incremental gain)

| arch | paper | exp39 | Δ |
|------|-------|-------|---|
| iKUN | 44.564 | 44.737 | +0.173 |
| FH V1 | 53.110 | 53.691 | +0.581 |
| FH V2 | 42.526 | 42.607 | +0.081 |

All beat paper but **same magnitude or smaller than depth-aug** — no incremental signal.

### Per-Class HOTA

| arch | depth APPEAR / MOVING / STATIC | exp39 APPEAR / MOVING / STATIC |
|------|-------------------------------|--------------------------------|
| iKUN | 46.79 / 32.07 / 44.86 | 46.79 / **29.35** / 43.98 |
| FH V1 | 55.84 / 45.48 / 49.73 | 55.78 / 45.15 / 49.45 |
| FH V2 | 41.96 / 48.69 / 45.17 | 41.72 / 48.51 / 45.32 |

**iKUN MOVING −2.72 HOTA** — appearance signal actively corrupts motion-class
discrimination at the cosine-similarity output. APPEAR class flat (CLIP visual contributes
nothing on top of bbox state slots [10:13] which already carry appearance signal per
project_gmc_is_motion_plus_bbox_specialist).

### Decision Gate Trace

- AUC: 0.7406 < 0.760 KILL → original Exp 39 kill (2026-05-05)
- HOTA: all 3 archs flat-NEG, max |Δ| = 0.229 — NOT ≥ +0.2 POS, NOT ≤ −0.5 strong NEG
- Per spec: no arch unlocks the depth-aug precedent → no n=3 retrain

## Why Did It Fail at HOTA Too?

Three factors compound:

1. **Cross-manifold corruption at MLP** — same root cause as AUC NEG. CLIP visual
   features (512D appearance manifold) concat'd into 13D motion forces the shared
   `motion_projector` MLP to disentangle motion ⊕ appearance dims while simultaneously
   aligning to language. Training cost is real (AUC −0.057) and survives downstream.

2. **Train-test bbox keying mismatch** — CLIP cache used at training is GT-keyed.
   At HOTA inference, tracker bboxes are NOT in cache; manager runtime-extracts CLIP
   B/32 on tracker crops. Tracker bboxes are noisier (drift, mismatched scale, occlusion
   gaps); CLIP forward produces a noisier appearance signal than the GT crops the aligner
   was trained on. Train-test distribution shift on the appearance manifold.

3. **Bbox state already carries APPEAR signal** — slots [10:13] = (cx_n, cy_n, w_n, h_n)
   are 13D's APPEAR proxy. Adding raw CLIP appearance to a vector already carrying spatial
   context is redundant for the APPEAR class (flat across all 3 archs) and harmful for
   MOVING (iKUN −2.72) where the appearance manifold drowns the velocity signal.

The motif is consistent with input-concat exhaustion (Exp 39 AUC, Exp 40, Exp 41):
**aligner-internal fusion sites cross-corrupt at the MLP**, regardless of fusion site
(input vs late) or modality (CLIP-visual vs CLIP-text). Decision-level fusion (logit+logit)
remains the only POS path for appearance signal.

## Code Status

All Exp 39 HOTA support stays merged on branch `exp/ego-motion-systematic`:
- `gmc_link/manager.py` — `use_clip_feat`/`clip_feat_dim`/`clip_proj_dim` ckpt-meta read,
  lazy CLIP B/32 load, `encode_clip_image_bboxes` helper, runtime extraction in
  `process_frame`.
- `run_build_gmc_cache_flexhook_v2_raw.py` — 2-pass builder threads `clip_feats_list`
  parallel to motions, passes through to `aligner.encode(clip_feats=...)`.
- 10 caches `gmc_link/gmc_scores_*_exp39_clip128_cache.json` (~84MB total).

Code dormant but functional. Future CLIP-feat experiments (e.g., decision-level CLIP-logit
fusion at iKUN/FlexHook output) can reuse the runtime extraction path.

## Decision

- **No new ship recipe.** Depth-aug iKUN ship (mean 44.823 ± 0.04 sample, +0.215 sig)
  remains live.
- **AUC-only-kill was right** for Exp 39 — this HOTA revisit confirms the original kill,
  contra the depth-aug precedent. AUC NEG → HOTA NEG holds when the failure is
  cross-manifold corruption, not just stage1 noise.
- **14 levers exhausted now ×2** at fusion-site (Exp 39 input-concat, Exp 40 cliptext
  joint, Exp 41 late-concat) — all aligner-internal fusion sites NEG. Decision-level
  fusion remains the only validated POS path for appearance signal.

## Lessons

- **AUC-NEG-survives-HOTA is depth-aug-specific** — that result was from adding
  ego-Z-compensated metric depth scalars (slots [13:16]: z_n, dz_residual_×3) to
  motion already in the manifold, not adding a foreign 512D appearance manifold.
- **Train-test keying matters** — GT-keyed train cache + tracker-keyed HOTA inference
  introduces an appearance distribution shift on top of the cross-manifold problem.
  Future appearance-fusion experiments should match training and inference key spaces.
- **iKUN MOVING is the canary** — when an aligner change regresses MOVING ≥ 2 HOTA at
  iKUN, the fusion-site change is corrupting motion gradient. APPEAR/STATIC stayed flat.
- 15 levers exhausted at stage1 ceiling 0.7793 micro AUC (now: depth-aug, world-XY,
  Exp 39 HOTA all closed). Pipeline is appearance-incompatible at any
  aligner-internal fusion site.
