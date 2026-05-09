# Depth-Augmented GMC (Approach B) — Design Spec

**Date:** 2026-05-09
**Status:** Design approved. Implementation pending.
**Brainstorm context:** 16 prior aligner-side levers exhausted (features, encoder, fusion site, curriculum, specialist aligners, OMF, zoned-flow, posthoc gate, cascade swap). All worked in 2D image plane. This spec opens the genuinely untouched lever class — 3D scene geometry via pseudo-depth.

## Hypothesis

The 13D motion vector `[res_dx×3, res_dy×3, dw, dh, cx_n, cy_n, w_n, h_n, snr]` is pure 2D image plane. KITTI is dashcam driving — depth IS the dominant scene gradient — but every track is currently blind to Z. Refer-KITTI test contains directional-distance expressions that 2D cannot disambiguate:

- V2 has 21 relative-construction expressions like `"the-automobiles-in-front-of-ours"`, `"vehicles-directly-in-front-of-you"` — pure Z semantics.
- V1 has 44 relative-construction expressions including `"counter-direction-vehicles-in-the-left"`.
- APPEAR class is the smallest GMC gain across all 3 archs (iKUN +0.04, V1 +0.18, V2 +0.21) — appearance + Z disambiguates `"the white car in front"` vs `"white cars far behind"`.
- Motion-class Δ saturates (V1 +1.804, V2 +0.740 multi-seed) — 2D residual velocity is squeezed dry; dZ adds a new approach/recede axis.

Depth ⊕ multi-scale dZ is hypothesized to break the 0.7793 micro AUC ceiling that 16 prior levers couldn't.

## Method

### Architecture

Extend 13D → 17D:

```
existing 13D                                      new 4D (depth)
[res_dx×3, res_dy×3, dw, dh, cx_n, cy_n, w_n, h_n, snr] ⊕ [Z_n, dZ_2, dZ_5, dZ_10]
                                                          = 17D
```

- `Z_n`: per-track depth at current frame, normalized `Z_meters / 100` (KITTI typical 0–80m, 100 chosen as headroom).
- `dZ_2 / dZ_5 / dZ_10`: `Z[t] − Z[t−gap]` at `FRAME_GAPS = [2, 5, 10]` matching existing multi-scale convention. No new timing axis.

`MotionLanguageAligner.motion_projector` input layer changes 13 → 17. Identity-init gate: zero-init the new 4 weight columns `motion_projector.weight[:, 13:17]` so step-0 forward is bit-exact w.r.t. existing 13D aligner. Trainable; gradient flows once training starts.

Rest of pipeline unchanged: `lang_projector`, fusion equation `fused = cs + b + α·(gmc−0.5)·sc`, per-arch ship recipes, EMA buffers.

### Depth Source

**Depth Anything V2 metric-vkitti large** via HuggingFace `transformers`:

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-VKITTI-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-VKITTI-Large-hf")
```

Reasoning:
- Metric output in meters — direct use, no scale ambiguity.
- vKITTI fine-tuned — distribution match for KITTI dashcam frames.
- ViT-L inference ~30 ms/frame on H100; one-shot pre-compute.

Fallback: `Metric3D-ViT-L` if depth-anything-v2-metric-vkitti out-of-distribution behavior surfaces on real KITTI frames during smoke test.

### Per-Track Z Sampling

For each frame, after detection bboxes are known:

1. Run depth model on full frame → metric depth map `D[H, W]`.
2. For each detection bbox `(x, y, w, h)`:
   - Sample 5×5 patch centered at `(x + w/2, y + h/2)`.
   - Take robust **median** over patch pixels.
   - Fallback to single-pixel sample if patch out-of-frame.
3. Store `Z[track_id, frame_id]` indexed by tracker output `predict.txt`.

Robust median (not mean) protects against bbox center landing on a sliver of background depth.

### Cache Strategy

**Per-track Z time-series cache. NOT raw depth maps.**

```
gmc_link/depth_cache/
  z_track_{arch}_{seq}.json
```

Format:
```json
{"<track_id>": {"<frame_id>": <z_meters>}}
```

Size estimate: ~50 tracks × ~600 frames × 8 bytes ≈ 0.25 MB/seq. Total: ~5 MB/arch × 3 archs × 22 seqs ≈ 330 MB. Trivial.

Raw depth maps NOT persisted (75 GB not stored).

### Ego-Z Compensation

KITTI ego-vehicle is moving forward → all stationary tracks' Z shrinks together. Naive `dZ_track` confounds object-Z-motion with ego-Z-motion.

**Compensation:** for each frame, compute `dZ_ego` as the median `dZ` over **all stationary tracks** (proxy: tracks with low 2D residual velocity, threshold `||res_v|| < 1 px/frame`). Then `dZ_residual = dZ_track − dZ_ego`. Stationary tracks should have `dZ_residual ≈ 0`; cars approaching/receding from ego frame stand out.

Edge case: if zero stationary tracks in frame (all moving), set `dZ_ego = 0` (skip compensation; raw `dZ_track` already informative when scene is dynamic).

### Normalization

| Feature | Range (raw) | Normalization | Range (norm) |
|---|---|---|---|
| `Z_n` | 0–80 m typical | `Z / 100` | 0–0.8 |
| `dZ_2` | ±5 m typical | `dZ / 10` | ±0.5 |
| `dZ_5` | ±10 m typical | `dZ / 10` | ±1.0 |
| `dZ_10` | ±20 m typical | `dZ / 10` | ±2.0 |

Matches existing 13D scale (cx_n ~0–1, residual_velocity ~±1 after VELOCITY_SCALE=100). Avoids feature dominance.

Clip `Z` to `[0, 80]` before normalization — Depth Anything V2 noise on far Z (>50 m) can produce outliers; saturate.

### Training Recipe

- Stage1: 100 ep, batch 256, lr 1e-3, MiniLM lang encoder, InfoNCE + FNM, temperature 0.07.
- 3 seeds (matches existing multi-seed protocol).
- Input: 17D motion + 384D lang → 256D shared embed.

## Files To Modify

### Create

- `gmc_link/depth_extractor.py` — Depth Anything V2 wrapper (`extract_depth_map(frame) → np.ndarray[H,W]`, batch helper).
- `gmc_link/depth_cache.py` — per-track Z cache build/load utilities (mirror `gmc_link/clip_cache.py` API).
- `run_build_depth_cache.py` — driver: iterate tracker `predict.txt` × seq, compute Z time-series, write cache.
- `run_build_gmc_cache_depth.py` — eval-time gmc score cache builder using 17D-aligner.

### Modify

- `gmc_link/manager.py` — `GMCLinkManager.process_frame` accepts optional `depth_z_lookup: dict[track_id, float]`. When present, append 4D depth features to motion vector (else 13D backward-compat path retained).
- `gmc_link/dataset.py` — `RMOTDataset` loads depth cache by `(arch, seq)` key, emits 17D motion vector when `--use-depth` flag set. Backward-compat default off.
- `gmc_link/alignment.py` — `MotionLanguageAligner.__init__` accepts `motion_dim` kwarg (default 13, set 17 when depth on). Identity-init the new 4 weight columns when `motion_dim=17` extending an existing 13D run.
- `gmc_link/train.py` — `--use-depth` flag, `--depth-cache-path` flag, plumb `motion_dim` through model construction + checkpoint metadata.
- `diagnostics/diag_gt_cosine_distributions.py` — read `motion_dim` from checkpoint metadata, pass to model construction. Backward-compat default 13.

### NO change to

- `gmc_link/losses.py` — no depth touchpoint.
- `gmc_link/text_utils.py` — lang side unchanged.
- Per-arch eval scripts (`run_ikun_linear_additive.py`, `run_flexhook_phase5_gmc_sweep.py`, `run_flexhook_v2_raw_sweep.py`) — read gmc cache only; recipe params unchanged.

## Verification Plan

### Step 0 — Smoke test depth extractor (1 frame)

Run Depth Anything V2 metric-vkitti large on KITTI seq 0011 frame 000100. Verify output is `[H, W]` float32 in meters, plausible range (3–80 m for KITTI street scene). Sample bbox center at known car location → assert Z ∈ [5, 60]. KILL if pipeline broken or Z is patently wrong (Depth Anything output mode mismatch).

### Step 1 — Build depth cache for 1 seq × 1 arch (smoke)

Run `run_build_depth_cache.py --arch ikun 0011`. Verify cache file written, ~50 tracks × ~600 frames coverage, no NaN/Inf. Spot-check 3 random `(track_id, frame_id)` pairs against bbox + raw depth viz.

### Step 2 — Identity gate (bit-exact at init)

Train 17D aligner with `motion_projector.weight[:, 13:17] = 0`. At step 0, forward batch through new aligner and existing 13D aligner. Assert `torch.allclose(scores_17d_init, scores_13d_init, atol=1e-5)`. Plumbing-bug detector.

### Step 3 — Stage1 train + 3-seq diag eval (decision gate)

Train 3 seeds, evaluate via `diagnostics/diag_gt_cosine_distributions.py` on V1 test 0005+0011+0013 pooled.

| Outcome | Action |
|---|---|
| micro AUC < 0.760 | KILL, retrospective. Depth doesn't help at aligner-internal level. |
| micro AUC ∈ [0.760, 0.7793) | Marginal. Run extended 5-seed; document. |
| micro AUC ∈ [0.7793, 0.79) | First-ever lever to match stage1 ceiling at 17D scale. Proceed to Step 4. |
| micro AUC ≥ 0.79 | Strong POS. First lever to break the ceiling. Prioritize Step 4 + 5. |

### Step 4 — HOTA cross-arch (3 archs single-seed first)

Build depth-routed gmc cache for iKUN / FH V1 / FH V2 at locked Arm A recipes. Eval pool HOTA single-seed.

| Per-arch outcome | Action |
|---|---|
| Pool Δ ≥ +0.05 vs Arm A multi-seed | Multi-seed n=3 confirm |
| Pool Δ ∈ [−0.05, +0.05] | NEUTRAL — depth helps separation but not HOTA; document |
| Pool Δ < −0.05 | NEG that arch; check if recipe-resweep `(α, sc, thr)` recovers (gmc distribution shifts under depth-augmented aligner) |

### Step 5 — Multi-seed ship gate

n=3 seeds × 3 archs at peak recipe.

| Pool Δ vs Arm A | Status |
|---|---|
| ≥ 2σ Arm A multi-seed (V1 ≥ +0.136, V2 ≥ +0.094, iKUN ≥ +0.048) | Ship for that arch |
| In [+0.05, +2σ) | Marginal; n=5 |
| < +0.05 | NEUTRAL/NEG; don't ship for that arch |

### Step 6 — Ablation: depth without ego compensation

Re-run Stage1 with `dZ_ego` zeroed (raw `dZ_track` only). Measures whether ego-Z compensation is the lever or raw depth alone is enough.

### Step 7 — Retrospective

`docs/superpowers/specs/2026-05-09-depth-augmented-gmc-retrospective.md` and memory entry `project_depth_augmented_gmc_{positive,negative}.md`.

## Risk Register

| Risk | Mitigation |
|---|---|
| Depth Anything V2 noise on far Z (>50 m) produces outliers | Clip Z ∈ [0, 80] before norm; saturation absorbs noise |
| dZ ego-Z confounding (forward ego → all dZ shifts) | Step 6 ablation isolates this. Ego compensation via stationary-track median |
| 17D corrupts existing 13D signal (cross-manifold corruption — same failure mode as Exp 39 input-concat NEG) | Identity-init the new 4 weight cols → bit-exact at step 0; gradient grows depth contribution only if useful |
| Depth Anything inference cost ~30 ms/frame × ~30k frames | One-shot pre-compute ~6 hr; cached per-track Z forever |
| KITTI seq intrinsic variation breaks absolute-Z meaning | If raw `Z_n` underperforms, fallback: `Z / Z_track_initial` (relative-to-first-frame depth ratio); Step 2 single-seq probe decides |
| Depth model OOD on certain seqs (rain/dusk) | KITTI tracking train+test all are clear daylight per dataset doc; risk minimal |
| `motion_dim=17` ckpt incompatible with `motion_dim=13` legacy ckpts | Checkpoint metadata stores `motion_dim`; loader branches; no silent reuse |

## Cost

- Depth pre-compute (3 archs × 22 seqs × ~30 ms/frame): ~6 hr GPU
- Stage1 train (3 seeds): ~6 hr GPU
- Cache build 3 archs (gmc scores at 17D aligner): ~3 hr GPU
- Eval + sweep: ~2 hr GPU
- **Total: ~17 hr GPU** to KILL/POS decision

## Reference Files

- 13D motion vector spec: `gmc_link/manager.py:GMCLinkManager.process_frame`
- Existing aligner: `gmc_link/alignment.py:MotionLanguageAligner.__init__`
- CLIP cache pattern (template for depth_cache.py): `gmc_link/clip_cache.py`
- Per-track tracker output: `NeuralSORT/{seq}/{car,pedestrian}/predict.txt`
- Multi-seed baseline (Arm A ship): `project_ikun_multiseed_positive.md`, `project_flexhook_multiseed.md`
- Depth Anything V2 model card: `huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large-hf`
- 16 prior aligner-side lever exhaustion: see `MEMORY.md` Project section
- Cross-manifold corruption precedent (Exp 39 input-concat NEG): `project_exp39_clip_concat_negative.md`

## Comparison Baselines

| Arch | Arm A multi-seed (n=3) | Paper | Depth-aug ship target (≥2σ) |
|---|---|---|---|
| iKUN | 44.608 ± 0.024 | 44.564 | ≥ 44.656 |
| FH V1 | 53.716 ± 0.068 | 53.110 (local B) / 53.824 (paper claim) | ≥ 53.852 |
| FH V2 | 42.799 ± 0.047 | 42.526 | ≥ 42.893 |
