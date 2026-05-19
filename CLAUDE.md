# CLAUDE.md

File guide Claude Code (claude.ai/code) working repo.

## Project Overview

GMC-Link = plug-and-play module **Referring Multi-Object Tracking (RMOT)**. Bridge object motion (geometry) + natural language (semantics). Input video + description like "moving cars", score which tracked objects match by physical motion reasoning, not visual appearance.

**Key Result**: +8.4% F1 gain (0.5730→0.6569) fused with iKUN using InfoNCE+FNM loss + learned fusion head.

## Common Commands

### Training

```bash
# Train the Motion-Language Aligner (main model)
python -m gmc_link.train

# Train the Fusion Head (3-stage pipeline)
python gmc_link/fusion_head.py --collect  # Step 1: collect iKUN logits + GMC scores
python gmc_link/fusion_head.py --train    # Step 2: train MLP
python gmc_link/fusion_head.py --eval     # Step 3: evaluate on validation split
```

### Inference & Evaluation

```bash
# Generate iKUN baseline predictions (vision-only, no GMC)
python run_ikun_baseline_video.py

# Generate fusion predictions (iKUN + GMC-Link)
python run_fusion_video.py --expr moving-cars

# Evaluate with HOTA metrics
python run_hota_eval.py                  # both methods
python run_hota_eval.py --method baseline
python run_hota_eval.py --method fusion

# Multi-expression demo inference
python gmc_link/demo_inference.py --multi
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
- `MotionLanguageAligner`: default = `shared_weight` arch (adopted 2026-05-19). Per-modality Linear adapter (motion 13→256, lang 384→256) → shared 2-hidden MLP (256→512→512→256) → LN → L2-norm. Symmetric two-tower, shared nonlinear core enforces common geometry by construction.
- Legacy `mlp` arch (independent dual-MLP, kept for backward-compat with pre-2026-05-19 checkpoints): `--architecture mlp`. Per-class HOTA equivalent at simple-fusion regime, dropped only for simplicity/principle.
- Inference: cosine similarity ∈ [-1, +1] = raw alignment score (no sigmoid/EMA per `feedback_simplicity_over_tiny_hota` Rule 1)
- Train symmetric InfoNCE loss + False-Negative Masking (`gmc_link/losses.py`)
- Language embeddings: SentenceTransformer (all-MiniLM-L6-v2, 384D) via `gmc_link/text_utils.py`

**Stage 4 — Fusion Head** (`gmc_link/fusion_head.py`):
- Tiny MLP: `[ikun_logit, gmc_score, is_motion_flag]` → 3→32→16→1 (sigmoid)
- `is_motion_flag`: 1.0 motion expressions, 0.5 stationary, 0.0 appearance-only
- Replace hand-tuned `min(vision, kinematic)` heuristic with learned combo

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
MotionLanguageAligner ←── TextEncoder("moving cars") → 384D embedding
    ↓
Cosine similarity score ∈ [0, 1]
    ↓
FusionHead([ikun_logit, gmc_score, is_motion_flag]) → P(match)
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
- EMA alphas: `MotionBuffer(α=0.3)`, `ScoreBuffer(α=0.4)`
- Embedding dims: motion 13D → 256D (Linear adapter), language 384D → 256D (Linear adapter) → shared trunk 256→512→512→256 (shared_weight arch)
- Fusion Head arch: 3→32→16→1 sigmoid output

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
- **Motion keyword detection**: ~38 motion keywords (moving, turning, parking, etc.) determine `is_motion_flag` in fusion head
- **Not for temporal trackers**: GMC-Link designed for spatially-ignorant vision-language frameworks (e.g., TransRMOT, iKUN). Cascading onto trackers with native temporal memory (e.g., TempRMOT) cause structural regression from redundant temporal constraints

## Experiment Log

Detailed experiment history (Exp 1–24+) in `RESEARCH_NOTES.md`, including ablations, loss comparisons, arch decisions with exact metric values.