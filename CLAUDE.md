# CLAUDE.md

File guide Claude Code (claude.ai/code) working repo.

## Project Overview

GMC-Link = plug-and-play module **Referring Multi-Object Tracking (RMOT)**. Bridge object motion (geometry) + natural language (semantics). Input video + description like "moving cars", score which tracked objects match by physical motion reasoning, not visual appearance.

**Key Result (2026-05-21 ship)**: 3-arch cross-validation, n=3 multi-seed, 3-seq pooled HOTA on V1 (V2 = 4-seq pooled). Sw aligner + per-arch linear additive fusion (raw cos, no EMA):
- iKUN: 44.634 ± 0.066 (+0.070 vs paper 44.564)
- FlexHook V1: 53.526 ± 0.087 (V1 paper-gap structural)
- FlexHook V2: 42.807 ± 0.038 (+0.281 vs paper 42.526)

Paper-beat 2/3. Earlier +8.4% F1 result (0.5730→0.6569 with learned MLP fusion head) HISTORICAL — F1-optimized head crashed HOTA, replaced by linear additive ship.

## Common Commands

### Training (Ship Aligner)

```bash
# Train shared_weight aligner (ship arch, seeds 0/1/2 for multi-seed)
for s in 0 1 2; do
  python -m gmc_link.train --split v1 --stage 1 \
      --architecture shared_weight --seed $s \
      --save-path gmc_link_weights_v1train_sharedweight_seed${s}.pth
done

# Legacy mlp arch (default; was prior ship arch until 2026-05-21)
python -m gmc_link.train --split v1 --stage 1 --architecture mlp

# Legacy F1-optimized fusion head (NOT ship; crashes HOTA — see project memory)
python gmc_link/fusion_head.py --collect / --train / --eval
```

### Ship Evaluation (Decision-Level Linear Additive Fusion)

```bash
# Build GMC caches per-arch per-seed (raw cosine, no EMA — GMC_RAW_COS=1 implied)
GMC_WEIGHTS=gmc_link_weights_v1train_sharedweight_seed0.pth \
GMC_SUFFIX=_sharedweight_seed0_rawcos GMC_RAW_COS=1 \
    python run_build_gmc_cache.py 0005          # iKUN cache
GMC_WEIGHTS=... GMC_SUFFIX=... GMC_RAW_COS=1 \
    python run_build_gmc_cache_flexhook.py 0005  # FH V1 cache
GMC_WEIGHTS=... GMC_SUFFIX=... GMC_RAW_COS=1 \
    python run_build_gmc_cache_flexhook_v2_raw.py 0005  # FH V2 cache

# Ship HOTA (n=3 seeds, locked recipes)
GMC_SUFFIX=_sharedweight_seed${N}_rawcos GMC_RAW_COS=1 \
    python run_ikun_linear_additive.py \
        --alpha 1.0 --gmc_scale 0.9  --thr 0.17 \
        --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10

GMC_SUFFIX=_sharedweight_seed${N}_rawcos GMC_RAW_COS=1 \
    python run_flexhook_phase5_gmc_sweep.py \
        --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
        --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9

GMC_SUFFIX=_sharedweight_seed${N}_rawcos GMC_RAW_COS=1 \
    python run_flexhook_v2_raw_sweep.py \
        --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
        --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2
```

### Ablation Studies

```bash
# Run structured ablation (multi-config evaluation)
python run_ablation_study.py

# Shell wrapper for batch ablation runs
bash run_ablation_proper.sh
```

### Package Installation

```bash
pip install -e .
# Dependencies: torch, torchvision, numpy, opencv-python, sentence-transformers, tqdm, scipy
```

## Architecture

### Pipeline Stages

**Stage 1 — Ego-Motion Compensation** (`gmc_link/core.py`):
- `ORBHomographyEngine` extracts ORB features, matches BFMatcher (Hamming, Lowe's ratio=0.7), RANSAC homography estimate
- Foreground mask prevent tracking object features instead static background
- Output: 3×3 homography matrix map prev frame → current frame

**Stage 2 — Cumulative Homography & Velocity** (`gmc_link/manager.py`):
- `GMCLinkManager` store *original* (never-warped) centroid coords in history deques
- Hold cumulative composed homographies: H[t-k→t] = H[t-1→t] @ ... @ H[t-k→t-k+1]
- Compute **multi-scale residual velocity** at three temporal gaps (2, 5, 10 frames) catch different motion patterns
- Residual velocity = raw velocity − ego velocity, isolate true object movement
- EMA smoothing: `MotionBuffer` (α=0.3) + `ScoreBuffer` (α=0.4) in `utils.py`
- Output **13D motion vector**: `[res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l, dw, dh, cx, cy, w, h, snr]`

**Stage 3 — Motion-Language Alignment** (`gmc_link/alignment.py`):
- `MotionLanguageAligner`: ship = `shared_weight` arch (2026-05-21 ship adoption). Per-modality Linear adapter (motion 13→256, lang 384→256) → shared 2-hidden MLP (256→512→512→256) → LN → L2-norm. Symmetric two-tower, shared nonlinear core. Trained `--architecture shared_weight`.
- Legacy `mlp` arch is code default (`--architecture mlp`): independent dual-MLP per modality (motion 13→256→512→256, lang 384→256→512→256) → L2-norm. Asymmetric per-modality projectors. Prior ship arch until 2026-05-21.
- Inference (ship): raw cosine (no sigmoid, no EMA) via `GMC_RAW_COS=1`. `manager.py:587` bypasses both cosine_buffer + sigmoid when raw_cos=True.
- Legacy inference (mlp ship era): sigmoid + EMA smoothing.
- Train symmetric InfoNCE loss + False-Negative Masking (`gmc_link/losses.py`)
- Language embeddings: SentenceTransformer (all-MiniLM-L6-v2, 384D) via `gmc_link/text_utils.py`

**Stage 4 — Decision-Level Linear Additive Fusion** (`run_ikun_linear_additive.py`, `run_flexhook_phase5_gmc_sweep.py`, `run_flexhook_v2_raw_sweep.py`):
- Ship formula: `final = model_logit + α · (sc · raw_cos + thr)` per arch per axis (motion + appearance)
- Per-arch recipe encodes (1) score-scale calibration (iKUN logits ~[0,1] vs FH ~[−10,+10] → different sc), (2) per-class GMC-relevance damping (sc_a 7-11× smaller than sc_m because GMC = motion signal is noise on appearance exprs).
- 18 free hyperparams (α/sc/thr × motion+appear × 3 archs). Auto-derive via std-matching = NEG (variant B falsified 2026-05-21).
- Legacy `gmc_link/fusion_head.py`: F1-optimized MLP `[ikun_logit, gmc_score, is_motion_flag]` → 3→32→16→1. NOT ship — crashes HOTA (−3.79 pool per `project_flexhook_learned_fusion_negative`).

### Data Flow

```
Video Frames
    ↓
ORBHomographyEngine → frame-to-frame H matrices
    ↓
GMCLinkManager → compose cumulative H, warp original coords, compute multi-scale residual velocity
    ↓
13D motion vector [res_dx×3scales, res_dy×3scales, dw, dh, cx, cy, w, h, snr]
    ↓
MotionLanguageAligner (shared_weight) ←── TextEncoder("moving cars") → 384D embedding
    ↓
raw cosine ∈ [−1, +1]   (GMC_RAW_COS=1; no sigmoid, no EMA)
    ↓
Per-arch linear additive fusion: final = model_logit + α · (sc · raw_cos + thr)
    ↓
HOTA-eval (TrackEval per-arch consumer: iKUN / FH V1 / FH V2)
```

### Training Data Pipeline (`gmc_link/dataset.py`)

- Load Refer-KITTI V2 expressions + ground-truth centroid tracks
- Multi-scale frame gaps `[2, 5, 10]` match `GMCLinkManager.FRAME_GAPS`
- Apply synthetic positional jitter (±2px) for robustness
- Normalize velocity: `v_norm = (v_pixel / img_dims) × 100` (resolution-invariant)
- Generate positive (motion_vector, language_embedding) pairs for InfoNCE train

### Key Constants

- `VELOCITY_SCALE = 100` (`utils.py`) — multiplier normalized velocities so MLP inputs ~1.0 magnitude
- `FRAME_GAPS = [2, 5, 10]` (`manager.py`) — must match between `GMCLinkManager` + `dataset.py`
- InfoNCE temperature: `0.07` (`losses.py`)
- EMA alphas: `MotionBuffer(α=0.3)`, `ScoreBuffer(α=0.4)`, `cosine_buffer(α=0.4)` — ship bypasses cosine_buffer via `GMC_RAW_COS=1`
- Embedding dims (ship `shared_weight`): motion/lang 13D/384D → 256D (Linear adapter) → shared trunk 256→512→512→256. Legacy `mlp`: motion 13D → 256D → 512D → 256D, language 384D → 256D → 512D → 256D.
- Ship recipes (per arch, locked):
  - iKUN: motion (α=1.0, sc=0.9, thr=+0.17) + appear (α=1.0, sc=0.30, thr=+0.10)
  - FH V1: motion (α=0.65, sc=10, thr=+3) + appear (α=1.0, sc=3.5, thr=+0.9)
  - FH V2: motion (α=0.4, sc=10, thr=+1.3) + appear (α=1.0, sc=3.5, thr=+1.2)
- Legacy Fusion Head arch (NOT ship): 3→32→16→1 sigmoid output

### Project Layout Notes

- `gmc_link/` — installable package (core library)
- `run_*.py` — top-level experiment/eval scripts (not in package)
- `build/` — stale `setuptools` build artifacts; do not edit
- Weight files (`*.pth`) + data files (`*.npz`) gitignored

## Data Paths

- Refer-KITTI dataset: `/home/seanachan/data/Dataset/refer-kitti-v2` (also symlinked `refer-kitti/` + `Refer-KITTI/`)
- Full annotation JSON: `Refer-KITTI_labels.json`
- iKUN precomputed scores: `iKUN/`
- NeuralSORT track detections: `NeuralSORT/`
- **GT template — TWO conventions, must pick right one (corrected 2026-04-30):**
  - `gt_template_old/` = **paper-iKUN-canonical**. Frame numbering aligns with NeuralSORT tracker `predict.txt`. Reproduces paper README 44.56 HOTA at 44.224 (cascade+simcalib YOLOv8-NS, 3-seq pooled). USE for any iKUN-paper comparison.
  - `gt_template/` = 2026-04-16 TransRMOT-convention regeneration. Frame numbering off-by-one vs NeuralSORT tracker. Using it drops HOTA ~6.4 due to gt-prediction misalignment (NOT a free eval improvement). Use only if pairing with TransRMOT-style tracker outputs.
  - Earlier note "fix closed ~10-point HOTA gap" was misleading — conflated the two label spaces. NeuralSORT tracker lives in `gt_template_old`'s convention.

## Important Design Decisions

- **ORB over optical flow**: ORB+Homography beat Farneback + RAFT on KITTI planar scenes; better outlier rejection via RANSAC
- **Decision-level fusion only**: Feature-level injection (motion into CLIP) caused catastrophic regression (−21.7% F1); always fuse at decision level
- **False-Negative Masking**: Multiple train samples share same expression; FNM prevent same-sentence pairs penalized as negatives
- **Cumulative homography**: Store original coords, warp once with composed H — more numerically stable than iterative per-frame warp
- **Multi-scale temporal velocity**: Three frame gaps (2, 5, 10) capture short/mid/long motion patterns; dominant ablation gain (+0.047 separation)
- **SNR feature**: Signal-to-noise ratio no improve mean separation but cut variance (±0.010 → ±0.007), stabilize predictions
- **Motion keyword detection**: ~38 motion keywords (moving, turning, parking, etc.) determine class for per-axis fusion in linear additive ship
- **Not for temporal trackers**: GMC-Link designed for spatially-ignorant vision-language frameworks (e.g., TransRMOT, iKUN). Cascading onto trackers with native temporal memory (e.g., TempRMOT) cause structural regression from redundant temporal constraints
- **Per-class GMC-relevance damping (2026-05-21)**: ship recipe sc_a (appear axis) is 7-11× smaller than sc_m (motion axis) per arch. GMC = motion signal is NOISE on appearance exprs ("black cars"). Hand-tuned damping suppresses this. Auto-deriving via std-matching falsified (variant B, all 3 archs NEG, see `project_variant_b_std_matching_negative_2026_05_21`).
- **Learned fusion heads = NEG**: F1-optimized MLP fusion head (`fusion_head.py`) crashes HOTA (−3.79 pool). Residual additive MLP on iKUN NEG. Hand-tuned linear additive strictly safer.

## Experiment Log

Detailed experiment history (Exp 1–24+) in `RESEARCH_NOTES.md`, including ablations, loss comparisons, arch decisions with exact metric values.