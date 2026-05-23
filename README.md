# GMC-Link: Global Motion Compensation for Referring Multi-Object Tracking

A plug-and-play module that helps modern RMOT models to compensate camera's ego-motion when referring motion-related expressions.

## What It Does

GMC-Link answers the question: **"Given a video and a sentence like _'moving cars'_, which tracked objects match that description?"**

It bridges the gap between **object motion** (geometry) and **language** (semantics) by:

1. **Compensating for camera motion** so that only true object movement remains.
2. **Encoding that motion** into an 13D geometric spatio-temporal vector (`[dx_s, dy_s, dx_m , dy_m ,dx_l , dy_l ,dw, dh, cx, cy, w, h ,snr]`).
   The motion representation is designed to explicitly capture both kinematic behavior and spatial context:

   - Multi-scale velocity (s/m/l) improves robustness under different frame gaps and noise levels
   - (dw, dh) captures scale changes (e.g., approaching / receding objects)
   - (cx, cy, w, h) provides spatial context for handling parallax
   - snr measures motion reliability and suppresses noisy tracks
3. **Aligning motion with language** using a learned `shared_weight` aligner (two-tower, shared nonlinear core) to produce a raw-cosine match score.

The score is then combined with a downstream tracker's own logits via **decision-level linear additive fusion** (`final = model_logit + α·(sc·raw_cos + thr)`). Current ship validates across 3 architectures (n=3 multi-seed): iKUN 44.634 ± 0.066 (+0.070 vs paper 44.564), FlexHook V1 53.526 ± 0.087, FlexHook V2 42.807 ± 0.038 (+0.281 vs paper) — **2/3 beat their paper anchors**. See [Current Ship](#current-ship-2026-05-21--3-architecture-cross-arch-validation) below for the locked recipes.

> **Note (historical):** An earlier learned-fusion-head era reported a +8.4% F1 gain (0.5730 → 0.6569) fused with iKUN. That F1-optimized MLP head was later falsified — it crashes pooled HOTA (−3.79) — and is superseded by the linear additive fusion above. The fusion head is retained only as legacy code.

---

## Architecture & Pipeline

```text
Video Frame ──► GMC (Homography) ──► Motion Feature Extraction (13D) ──► shared_weight Aligner (InfoNCE) ──► Linear Additive Fusion with Tracker Score ──► Final Association
                                                                      ▲
Natural Language Prompt ──► SentenceTransformer Embedding ────────────┘
```

> We're training a Neural Network that can align the textual embeddings and motion embeddings together, and give a score of their alignment.

### Key Components

| Module                    | File                                  | Role                                                                                                                                                                                                      |
| ------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GlobalMotion**          | `core.py`                             | Detects camera movement via ORB feature matching and RANSAC homography estimation. Returns the homography matrix and background warp residual.                                                 |
| **Utilities**             | `utils.py`                            | `warp_points()` transforms previous positions into the current frame's coordinate system. `normalize_velocity()` makes velocities scale-invariant. `MotionBuffer` applies EMA smoothing to reduce jitter. |
| **MotionLanguageAligner** | `alignment.py`                        | `shared_weight` arch (ship default): per-modality Linear adapter (motion 13→256, lang 384→256) → shared 2-hidden MLP (256→512→512→256) → LN → L2-norm. Cosine similarity in shared 256D space. Legacy `mlp` arch (asymmetric dual-MLP) also supported via `--architecture mlp`. |
| **TextEncoder**           | `text_utils.py`                       | Wraps `all-MiniLM-L6-v2` (SentenceTransformers) to encode natural language prompts into 384-dim embeddings.                                                                                               |
| **GMCLinkManager**        | `manager.py`                          | The orchestrator. Maintains cumulative homographies, computes multi-scale ego-compensated residual velocities, and queries the aligner for alignment scores.                                                                 |
| **Decision-Level Fusion** | `run_ikun_linear_additive.py`, `run_flexhook_phase5_gmc_sweep.py`, `run_flexhook_v2_raw_sweep.py` | Per-arch linear additive fusion: `final = model_logit + α · (sc · raw_cos + thr)`. Ship over `fusion_head.py` (F1-optimized MLP head, crashes HOTA per project memory). |
| **Dataset & Training**    | `dataset.py`, `train.py`, `losses.py` | Builds (motion, language) training pairs from [Refer-KITTI V2](https://github.com/wudongming97/RMOT) using symmetric InfoNCE loss.                                              |
| **Demo Inference**        | `demo_inference.py`                   | End-to-end evaluation on iKUN + GMC-Link fusion across all expressions in a sequence.                                                                                                                      |

---

## How It Works (Step by Step)

1. **Feature-based GMC**: Between consecutive frames, ORB keypoints are matched on the _background_ (tracked objects are masked out). A homography matrix `H` is estimated via RANSAC to represent pure camera motion.

2. **Cumulative Homography**: Homographies are composed cumulatively (`H[t-k→t] = H[t-1→t] @ ... @ H[t-k→t-k+1]`). Original centroid coordinates are stored unmodified and warped once when computing velocity — more numerically stable than iterative warping.

3. **Residual Velocity**: For each tracked object, `residual_v = raw_v - ego_v` where `ego_v = warp(old_centroid, H) - old_centroid`. This subtracts camera motion, isolating true object movement. Computed at three temporal scales (gap=2, 5, 10 frames) to capture different motion patterns.

4. **13D Motion Vector**: `[res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l, dw, dh, cx, cy, w, h, snr]` — multi-scale residual velocity (6D), bbox changes (2D), spatial position (4D), and signal-to-noise ratio (1D).

5. **Language Encoding**: The user's text prompt (e.g., _"moving cars"_) is encoded once into a 384-dim vector using a SentenceTransformer.

6. **Alignment Scoring**: The `shared_weight` aligner (per-modality Linear adapter → shared 256→512→512→256 trunk → LN → L2-norm) projects the 13D motion vector and the 384-dim language vector into a shared 256-dim space. The **raw cosine similarity** (no sigmoid, no EMA — `GMC_RAW_COS=1`) is the GMC score fed into decision-level fusion. (A legacy sigmoid + EMA path producing a `[0, 1]` score is still available but is not the current ship.)

---

## Framework

### Loss Calculation

  $$
  \mathcal{L} = \frac{1}{2B} \sum_{i=1}^{B} \left[ -\log
  \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B} \exp(s_{ij}/\tau)} \;-\; \log
  \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B} \exp(s_{ji}/\tau)} \right]
  $$

   where:

- $s_{ij} = \hat{m}_i \cdot \hat{l}_j$ = cosine sim (motion embed i · language embed
  j)
- $\hat{m}, \hat{l}$ = L2-normed 256D embeds
- $\tau = 0.07$
- $B$ = batch size (256, Stage 1 default)
- diagonal $s_{ii}$ = positive pair

  First term = motion $\rightarrow$ language. Second term = language $\rightarrow$ motion.

  With FNM (mask same-sentence off-diagonals from denom):

  $$
  \mathcal{L}{m2l} = -\frac{1}{B}\sum_i \log \frac{\exp(s{ii}/\tau)}{\exp(s_{ii}/\tau)
  + \sum_{j \in \mathcal{N}i} \exp(s{ij}/\tau)}
  $$

  where $\mathcal{N}_i = {j : j \neq i, \text{sent}(j) \neq \text{sent}(i)}$

### Evaluation Metric

The project evaluates exclusively on **HOTA** (pooled, per downstream tracker). AUC is not used as a metric or gate — score separation at the aligner stage was decoupled from downstream HOTA, so HOTA is reported directly. See [Current Ship](#current-ship-2026-05-21--3-architecture-cross-arch-validation) for multi-seed HOTA results.

## Training

- **Dataset**: [Refer-KITTI](https://github.com/wudongming97/RMOT) — KITTI tracking sequences annotated with natural language expressions describing object motion.
- **Supervision**: Symmetric InfoNCE loss with False-Negative Masking (FNM). Positive pairs come from ground-truth matches; negatives are formed in-batch. FNM prevents same-sentence pairs from being treated as false negatives.
- **Motion keywords filtered**: Only expressions involving motion concepts (`moving`, `turning`, `parking`, `approaching`, etc.) are used — since the model only sees velocity vectors, not appearance.

---

## Usage

### Inference with Decision-Level Linear Additive Fusion (Current Ship)

The current ship fuses GMC-Link's **raw cosine** score with a downstream tracker's logit via a per-arch linear additive rule: `final = model_logit + α·(sc·raw_cos + thr)`. There is no learned fusion model at inference — only the locked (α, sc, thr) recipe per arch (see [Current Ship](#current-ship-2026-05-21--3-architecture-cross-arch-validation)). Reproduce via the per-arch sweep scripts:

```bash
# iKUN (cascade+simcalib, YOLOv8-NS)
GMC_RAW_COS=1 python run_ikun_linear_additive.py

# FlexHook V1 / V2
GMC_RAW_COS=1 python run_flexhook_phase5_gmc_sweep.py
GMC_RAW_COS=1 python run_flexhook_v2_raw_sweep.py
```

The raw GMC score for each track (consumed by the rule above) is produced by:

```python
import os
os.environ["GMC_RAW_COS"] = "1"  # raw cosine, no sigmoid/EMA — current ship
from gmc_link import GMCLinkManager, TextEncoder

encoder = TextEncoder(device="cuda")
linker = GMCLinkManager(weights_path="gmc_link_weights.pth", device="cuda", lang_dim=384)

language_embedding = encoder.encode("moving cars")
raw_cos, _ = linker.process_frame(frame, active_tracks, language_embedding)

# Per-arch linear additive fusion (iKUN motion-axis recipe shown)
alpha, sc, thr = 1.0, 0.9, 0.17
final_score = ikun_logit + alpha * (sc * raw_cos[track_id] + thr)
```

> **Legacy:** An earlier learned fusion head (`gmc_link/fusion_head.py`, `load_fusion_head`) combined `[ikun_logit, gmc_score, is_motion_flag]` via a 3→32→16→1 MLP. It was falsified (F1-optimized MLP crashes pooled HOTA −3.79) and is **not** the recommended path. Code is retained for reproducibility only.

### Standalone GMC-Link (without iKUN)

```python
encoder = TextEncoder(device="cuda")
linker = GMCLinkManager(weights_path="gmc_link_weights.pth", device="cuda", lang_dim=384)

language_embedding = encoder.encode("moving cars")
scores, velocities = linker.process_frame(frame, active_tracks, language_embedding)
# With GMC_RAW_COS=1 (ship default): scores = {track_id: 0.62, ...}  raw cosine ∈ [−1, +1]
# Legacy (sigmoid+EMA, GMC_RAW_COS unset): scores ∈ [0, 1]
```

### Training the Aligner

```bash
python -m gmc_link.train
```

### Training the Fusion Head (Legacy)

> The learned fusion head is **legacy** — superseded by the linear additive fusion above (the MLP head crashes pooled HOTA −3.79). Retained for reproducibility only.

```bash
python gmc_link/fusion_head.py --collect  # collect iKUN + GMC-Link training data
python gmc_link/fusion_head.py --train    # train the fusion MLP
python gmc_link/fusion_head.py --eval     # evaluate on validation split
```

### Multi-Expression Evaluation

```bash
python gmc_link/demo_inference.py --multi
```

---

## Ablation Study: Motion Vector Design

Progressive feature addition evaluated on seq 0011, expr "moving-cars" (score separation = GT avg − NonGT avg). 3 runs each for statistical reliability.

| Config | Dim | Features | Mean Sep | Std |
|--------|-----|----------|----------|-----|
| A: 8D no-ego | 8 | `[raw_dx, raw_dy, dw, dh, cx, cy, w, h]` | +0.344 | ±0.012 |
| B: 8D ego | 8 | `[res_dx, res_dy, dw, dh, cx, cy, w, h]` | +0.354 | ±0.031 |
| C: 12D multi-scale | 12 | `[res_dx×3scales, dw, dh, cx, cy, w, h]` | **+0.401** | ±0.010 |
| **D: 13D full** | **13** | **`[..., snr]`** | **+0.395** | **±0.007** |
| E: 10D raw+ego | 10 | `[raw_dx, raw_dy, ego_dx, ego_dy, dw, dh, cx, cy, w, h]` | +0.351 | ±0.029 |

**Key findings:**
- **Multi-scale temporal (B→C, +0.047)** is the dominant improvement — short/mid/long windows capture different motion patterns.
- **Ego compensation (A→B, +0.010)** provides a small improvement but high variance.
- **SNR (C→D)** doesn't improve mean separation but **reduces variance** (±0.010 → ±0.007), stabilizing predictions.
- **13D** is chosen as the final config for its best stability.

---

## Current Ship (2026-05-21) — 3-Architecture Cross-Arch Validation

GMC-Link plugs into 3 downstream RMOT consumers via decision-level linear additive fusion. Evaluated on Refer-KITTI V1 (3-sequence pooled HOTA: 0005, 0011, 0013) and V2 (4-sequence pooled: 0005, 0011, 0013, 0019), n=3 multi-seed.

### Ship Pipeline

```
final_score = model_logit + α · (sc · raw_cos + thr)
```

- **Aligner:** `shared_weight` (Linear adapter motion 13→256 + lang 384→256 → shared MLP 256→512→512→256 → LN → L2)
- **Aligner training:** V1 stage1, InfoNCE+FNM (τ=0.07), 100 ep, batch 256, lr 1e-3, seeds {0,1,2}
- **GMC score:** raw cosine ∈ [−1,+1] (no sigmoid, no EMA — `GMC_RAW_COS=1`)
- **Fusion:** per-arch (α, sc, thr) on motion + appearance axes

### Multi-Seed HOTA (n=3 mean ± sample std)

| arch | Raw Baseline (no GMC) | + GMC Ship | Δ vs Raw | Paper anchor | Δ vs Paper |
|---|---|---|---|---|---|
| **iKUN** (cascade+simcalib, YOLOv8-NS) | 44.224 | **44.634 ± 0.066** | +0.410 | 44.564 | **+0.070** |
| **FlexHook V1** (Temp-NeuralSORT-kitti1) | 53.110 | **53.526 ± 0.087** | +0.416 | 53.824 | −0.298 |
| **FlexHook V2** (Temp-NeuralSORT-kitti1, V2 labels) | 42.526 | **42.807 ± 0.038** | +0.281 | 42.526 | **+0.281** |

**Paper-beat count: 2/3** (iKUN +0.070, V2 +0.281). FH V1 paper-gap structural in all configurations tested.

### Per-Arch Recipes (locked)

| arch | α_m | sc_m | thr_m | α_a | sc_a | thr_a |
|---|---|---|---|---|---|---|
| iKUN | 1.0 | 0.9 | +0.17 | 1.0 | 0.30 | +0.10 |
| FH V1 | 0.65 | 10 | +3 | 1.0 | 3.5 | +0.9 |
| FH V2 | 0.4 | 10 | +1.3 | 1.0 | 3.5 | +1.2 |

The 18 hyperparams encode two effects: (1) per-arch score-scale calibration (model logits live in different ranges per backbone — iKUN [0,1], FH [−10, +10+]) + (2) per-class GMC-relevance damping (sc_a is 7-11× smaller than sc_m because GMC = motion signal is noise on appearance expressions like "black cars"). Auto-deriving sc via std-matching was tested and falsified (variant B, all 3 archs catastrophic NEG).

> **Note:** Feature-level injection of motion embeddings into iKUN's CLIP visual pipeline was also explored but causes catastrophic regression (−21.7% F1) because additive injection corrupts the CLIP representation. Decision-level fusion is the correct approach.

---

## TransRMOT Integration & Performance (Historical, pre-2026-05)

> **Note:** This section documents the earlier `min(vision_prob, kinematic_prob)` fusion era on TransRMOT. The current ship (above) uses per-arch linear additive fusion on iKUN/FlexHook V1/V2 instead. TransRMOT integration is retained as historical context.

### How It's Plugged In (For Developers)

Integrating GMC-Link into an existing tracker like TransRMOT is straightforward because GMC-Link acts as a **post-processing filter** on top of the tracker's own predictions.

Here is the step-by-step data flow of how GMC-Link was injected into TransRMOT's `inference.py` loop:

1. **Initialize the Manager:** We instantiate `GMCLinkManager` and `TextEncoder` alongside TransRMOT's core model. We encode the text prompt (e.g., "a red car moving left") once at the start of the video.
2. **Intercept Detections:** For every video frame, TransRMOT generates a list of associated bounding boxes. We intercept these boxes _before_ TransRMOT makes its final filtering decisions.
3. **Generate Kinematic Scores:** We pass the intercepted boxes and the current video frame into `GMCLinkManager.process_frame()`. GMC-Link computes the ego-motion, calculates the 13D velocity vectors, and asks its MLP aligner: _"Based purely on physics, how well do these boxes match the text prompt?"_ It returns a probability score between 0 and 1 for each box.
4. **Strict Minimax Fusion:** TransRMOT initially generates a "Vision Probability" (does this _look_ like a red car?). GMC-Link generates a "Kinematic Probability" (is this object _moving_ left?). We mathematically fuse them using a strict intersection: `final_score = min(vision_prob, kinematic_prob)`.
5. **Final Output:** If a stationary red car tricked TransRMOT's vision model, its `vision_prob` would be `0.9`. But GMC-Link's `kinematic_prob` would be `0.01` (because it's stationary). The `min()` function suppresses the score to `0.01`, instantly filtering out the hallucination.

**Example Code Integration (`inference.py`)**:

```python
# Inside TransRMOT's main evaluation loop
from gmc_link.manager import GMCLinkManager

gmc_linker = GMCLinkManager(weights_path="checkpoints/gmc_link.pth", device="cuda")

for frame in video_frames:
    # 1. TransRMOT native visual detection
    dt_instances = detector.detect(frame, text_prompt)

    # 2. Intercept and format for GMC-Link
    active_tracks = format_boxes_for_gmc(dt_instances)

    # 3. Geometric kinematic evaluation
    gmc_scores, _ = gmc_linker.process_frame(frame, active_tracks, language_embed)

    # 4. Strict Minimax Fusion
    for track in dt_instances:
        vision_prob = track.refers
        kinematic_prob = gmc_scores.get(track.track_id, 0.0)

        # Override vision hallucination with strict physical intersection
        track.refers = min(vision_prob, kinematic_prob)
```

### Benchmark Results

By enforcing this `min(vision_prob, kinematic_prob)` requirement during evaluation, GMC-Link securely grounded visual tracking with real-world spatial physics, destroying hallucinated trajectories while vastly elevating Association Accuracy (`AssA`).

| Tracker Configuration                | HOTA      | DetA      | AssA      | DetRe | DetPr |
| ------------------------------------ | --------- | --------- | --------- | ----- | ----- |
| **Baseline TransRMOT (Vision Only)** | 38.06     | 29.28     | 50.83     | 40.19 | 47.36 |
| **TransRMOT + GMC-Link (Ours)**      | **42.61** | **28.41** | **69.29** | 37.12 | 47.29 |

_In this historical `min(vision_prob, kinematic_prob)` era, integration produced a **+18.4% absolute surge** in Tracking Association and reached **`42.61` HOTA** on TransRMOT, demonstrating that geometry-aware fusion outperforms pure vision. Note: this `42.61` is a past min()-fusion result on TransRMOT and is **not** the current ship — the current ship uses per-arch linear additive fusion on iKUN/FlexHook V1/V2 (see [Current Ship](#current-ship-2026-05-21--3-architecture-cross-arch-validation))._

---

## TempRMOT Integration & Temporal Constraints

> **Note:** The experiments below used the historical `min(vision_prob, kinematic_prob)` fusion (not the current linear additive ship), but the conclusion — **do not cascade GMC-Link onto trackers with native temporal memory** — holds independently of the fusion rule and remains a valid design constraint.

### The Double-Tracking Problem

While GMC-Link drastically enhances models operating purely on spatial language (like TransRMOT), integrating GMC-Link into architectures featuring **native temporal memory** computationally causes a structural regression.

When evaluated dataset-wide across the dynamic motion corpus (136 sequences) inside `TempRMOT`—which natively caches 8-frame multi-head attention trackers out-of-the-box:

| Tracker Configuration | HOTA | DetA | AssA |
| --- | --- | --- | --- |
| **Baseline TempRMOT (Native 8-frame memory)** | **49.930** | **37.221** | **67.172** |
| **TempRMOT + GMC-Link (Thr: 0.4)** | 43.177 | 29.723 | 62.860 |

### Why Did Validation HOTA Drop?

Because TempRMOT outputs heavily smoothed, highly-confident bounding vectors using its native temporal engine, forcing our strict `min(vision_prob, kinematic_prob)` fusion upon it operates as a redundant, secondary physical constraint. Mathematically, this arbitrarily drags validly-tracked identities down below TempRMOT's absolute deletion boundary (`filter_dt_by_ref_scores(0.4)`), causing thousands of True Positives to permanently vanish.

#### Addendum: Threshold Ablation Study (`0011+moving-cars`)

To formally verify if adjusting TempRMOT's internal deletion boundary could recover the performance regression, we conducted a targeted ablation on the `0011+moving-cars` subset by manually relaxing the deletion floor from `0.4` down to `0.2` when fusing GMC-Link probabilities.

| Setup (`0011+moving-cars`)         | HOTA       | DetA       | AssA       |
| ---------------------------------- | ---------- | ---------- | ---------- |
| **Baseline TempRMOT (Thr: 0.4)**   | **39.896** | 24.664     | **64.502** |
| TempRMOT + GMC-Link (Thr: 0.4)     | 29.408     | 18.591     | 46.529     |
| **TempRMOT + GMC-Link (Thr: 0.2)** | **39.797** | **28.350** | 55.881     |

Lowering the deletion threshold to `0.2` **completely recovered** the catastrophic 10% subset HOTA regression, bringing metrics cleanly back to parity with the baseline (~39.8%). Relaxing the probability floor allowed statistically-suppressed tracking links to survive, proving that GMC-Link was strictly penalized by TempRMOT's rigid `0.4` validation boundary. Ultimately, this mathematically traded Association Accuracy (-8.6%) for pure Detection Accuracy (+3.68%).

> [!WARNING]
> **Developer Insight:**
>
> 1. GMC-Link is a state-of-the-art plug-and-play geometric filter mathematically designed to rescue **spatially-ignorant Vision-Language frameworks** (e.g., TransRMOT).
> 2. It **should not** be cascaded onto frameworks that independently construct recursive temporal bounding boxes natively (like TempRMOT/Refer-SORT). While unilaterally lowering the underlying model's deletion threshold computationally recovers the HOTA destruction, cascading redundant temporal tracking pipelines remains fundamentally structurally hostile.

---

## Key Design Decisions

- **Geometry over appearance**: GMC-Link reasons purely about _motion_, making it complementary to vision-language models that reason about _appearance_.
- **Plug-and-play**: Works with any tracker (ByteTrack, BoT-SORT, TransRMOT) — just provide track centroids.
- **Lightweight**: The aligner MLP is tiny (~few hundred KB), adding negligible overhead to an existing tracking pipeline.
