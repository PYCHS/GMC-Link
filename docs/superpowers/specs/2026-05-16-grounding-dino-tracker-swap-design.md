# Path A: Grounding-DINO + OC-SORT Tracker Swap — 2026-05-16

## Goal

Lift GMC-Link iKUN ship recipe by ≥+2.23 HOTA (5% relative) from current
multi-seed n=3 mean **44.608 ± 0.024** toward target ~46.8+. After 19-lever
ceiling-break campaign (score-side, aligner-internal, decision-fusion, all
NEG), the only remaining cost-bounded direction targets the actual bottleneck
identified by the failure audit: detector recall + tracker FN on the motion
class.

Paper's own +4.28 HOTA jump (YOLOv8-NS 44.564 → DDETR-NS 48.84) came entirely
from detector + tracker swap, NOT from cascade/fusion changes. DDETR-NS is
BLOCKED (paper author refused 3× to release tracker output; NeuralSORT code
itself unreleased). Path A reproduces that direction with publicly-available
components.

## Architecture

- **Detector:** Grounding-DINO-Tiny (open-vocab, Swin-T backbone) replaces
  YOLOv8-NS. Prompts: KITTI 8-class set ("car", "pedestrian", "bicyclist",
  "van", "truck", "motorcyclist", "bus", "train"). Threshold: 0.25 (paper
  default) for box, 0.20 for text — Grounding-DINO standard.
- **Tracker:** OC-SORT replaces NS. KF + observation-centric association,
  no Re-ID dependency, single-stage simple integration. Fallback: StrongSORT
  with OSNet Re-ID if OC-SORT underperforms on Phase A2 gate.
- **Downstream:** iKUN cascade + simcalib + locked appear-ship recipe
  (motion α_m=1.0 sc_m=0.9 thr_m=+0.17, APPEAR α_a=1.0 sc_a=0.30 thr_a=+0.10)
  unchanged. GMC-Link aligner unchanged. **No retraining**, this is a
  pre-iKUN component swap only.

## Decomposition into Phases

### Phase A1 — Detector Cache Build + Recall Gate (~3 days)

**Files to create:**

- `run_build_grounding_dino_cache.py` — load `IDEA-Research/grounding-dino-tiny`
  via HuggingFace transformers, iterate over `refer-kitti/KITTI/training/image_02/{seq}/*.png`,
  query per-class prompts, write `det_cache/grounding_dino_v1/{seq}/{class}/dets.json`
  matching existing DDETR schema (`{seq, class, img_h, img_w, score_thr, schema, frames: {fid: [[x1,y1,x2,y2,score], ...]}}`).
- KITTI seqs: 0005, 0011, 0013 (test split). Classes: car + pedestrian only
  (matching existing NeuralSORT subdir structure — other 6 classes have zero
  GT in V1 RMOT expressions per `gmc_link/expr_class.py`).

**Files unchanged:** Refer-KITTI GT, iKUN cascade weights, GMC ship recipe.

**Gate G1 — Detector Recall**

- For each (seq, class), compute box-recall against YOLOv8-NS det cache
  (`det_cache/DDETR-kitti/{seq}/{class}/dets.json` is DDETR — need to locate
  YOLOv8-NS cache equivalent or extract from NS tracker predict.txt).
  Wait: actual baseline detector is whatever feeds NS — for our ship we have
  NS tracker outputs in `NeuralSORT/{seq}/{class}/predict.txt`, which are
  POST-tracker. The ship's detector is *implicit* via NS predict.txt.
  → Use NS predict.txt boxes as "current pipeline reference" for G1 recall.
- Match Grounding-DINO box to NS box by IoU ≥ 0.5. Recall = matched / total NS.
- **PASS G1:** recall ≥ 0.90 per seq per class AND ≥10% boxes are NEW
  (Grounding-DINO box with no NS match — required, else net-neutral swap).
  Threshold 0.90 (not 0.95) accommodates open-vocab vs KITTI-trained domain
  gap; below 0.90 the swap can't compete on motion class FN.
- **FAIL G1:** Kill Path A. Conclude open-vocab detector cannot match KITTI
  domain. Document as 20th NEG lever. Consider Path C (LVLM rerank).

### Phase A2 — Tracker Integration + HOTA Gate (~5 days)

**Files to create:**

- `run_ocsort_on_grounding_dino.py` — OC-SORT inference using
  `pip install ocsort` (or vendored implementation). Input: Phase A1 detection
  cache. Output: `tracker_outputs/grounding_dino_ocsort/{seq}/{class}/predict.txt`
  in MOT format `frame,id,x,y,w,h,score,-1,-1,-1` (1-indexed frame, matching
  existing NS schema). OC-SORT params: det_thresh=0.30, iou_threshold=0.30,
  delta_t=3, max_age=30 (KITTI defaults).
- `run_ikun_linadd_grounding_ocsort.py` — fork of `run_ikun_linadd_botsort.py`
  (closest existing analog: same iKUN-linear-additive ship recipe, swappable
  tracker root). Read `tracker_outputs/grounding_dino_ocsort/` instead of
  `NeuralSORT/`, otherwise identical pipeline.

**Files unchanged:**

- iKUN cascade weights, GMC-Link aligner weights, ship recipe constants.
- Standard inference driver `run_ikun_baseline_video.py` left untouched;
  new driver isolated to its own file.

**Gate G2 — Single-Seed HOTA**

- Run full pipeline through cascade + GMC ship recipe. Output:
  3-seq pooled HOTA on `gt_template_old`.
- **PASS G2:** Δ ≥ +0.50 vs ship 44.586 (seed-1 reference).
- **MARGINAL G2:** Δ ∈ [+0.0, +0.50] → run multi-seed verify (Phase A3).
- **FAIL G2:** Δ < 0 → revert to NS pipeline, evaluate OC-SORT vs StrongSORT
  swap. If StrongSORT also <0 → conclude tracker quality is the binding
  constraint (consistent with ByteTrack/BoT-SORT precedent on DDETR dets);
  escalate to Path C.

### Phase A3 — Multi-Seed Validation (~2 days)

If G2 marginal or pass: 3-seed re-run of full pipeline (OC-SORT is
deterministic per fixed dets, but seed-vary the GMC-Link aligner inference
to stay consistent with prior multi-seed framework — n=3 same aligners we
already have cached).

**Gate G3 — Multi-Seed Significance**

- Paired-t (per-seq Δship over 3 V1 seqs × 3 seeds = 9 cells) vs 44.608.
- **PASS G3:** Δ ≥ +1.0 multi-seed mean AND p < 0.10.
- **PARTIAL G3:** Δ ∈ [+0.5, +1.0] → ship as new conservative recipe, then
  Phase A4.
- **FAIL G3:** Drop swap. Accept that public tracker can't match NS-paper.

### Phase A4 (conditional) — Stack with SAM2 + LVLM Rerank (~3 weeks)

ONLY if A3 ≥ +0.5 but < +2.23. Stack additional levers on top of
Grounding-DINO + OC-SORT base:

1. **SAM2 pixel-tight crops** → re-feed cascade. Tests if cascade APPEAR
   absorption fails on bbox-loose crops.
2. **Qwen2-VL-7B rerank** → 4th additive channel
   `fused = cs + b + α_m·gmc_m·sc_m + α_a·gmc_a·sc_a + α_lvlm·lvlm·sc_lvlm`.
   Sweep α_lvlm ∈ {0.5, 1.0}, sc_lvlm ∈ {1.0, 2.0}.

Each stacked lever has its own kill gate (Δ ≥ +0.3 to retain).

## Critical Files

**Phase A1:**
- Create: `run_build_grounding_dino_cache.py`
- Create: `det_cache/grounding_dino_v1/{seq}/{class}/dets.json`

**Phase A2:**
- Create: `run_ocsort_on_grounding_dino.py`
- Create: `tracker_outputs/grounding_dino_ocsort/{seq}/{class}/predict.txt`
- Create: `run_ikun_linadd_grounding_ocsort.py`

**Phase A3:**
- Reuse `run_paired_wilcoxon.py` or equivalent multi-seed t-test script.

**Phase A4 (conditional):**
- Path B/C specs follow if triggered (separate design doc).

## Risks + Mitigations

- **R1: Domain gap (Grounding-DINO COCO-pretrained vs KITTI driving).**
  Grounding-DINO recall on KITTI cars may underperform YOLOv8 (KITTI-trained).
  Mitigation: G1 gate kills cheaply (~3 days) if recall < 95%.
- **R2: Tracker FN floor (public tracker can't match NS).**
  Precedent: ByteTrack 39.0–39.8, BoT-SORT 35.12, both on DDETR dets, both <40
  (`project_path2_ddetr_public_trackers_negative`). OC-SORT may fall in same
  range. Mitigation: G2 gate kills at HOTA level; StrongSORT fallback before
  fully aborting.
- **R3: Cascade trained on YOLOv8-NS bbox distribution.**
  Cascade visual stream may not transfer if Grounding-DINO produces
  systematically tighter/looser boxes. Mitigation: full eval at G2 captures
  this; box-shape diff diagnostic part of G1 reporting.
- **R4: 5% target unreachable from single lever.**
  Honest scope: Phase A standalone (detector + tracker swap only) realistic
  range is +0.5 to +2.0 HOTA based on paper-internal DDETR/NS delta and
  public-tracker precedent. The +2.23 (5%) target almost certainly requires
  the A4 stack (SAM2 + LVLM on top). Spec budgets accordingly; if A1–A3
  alone hits +2.23 it would be unexpected upside.

## Decision Tree

```
A1: build dets → G1 recall gate
  G1 FAIL → kill, document, consider Path C
  G1 PASS → A2
A2: OC-SORT integration → G2 HOTA gate
  G2 FAIL (Δ<0) → try StrongSORT once; if also FAIL → escalate Path C
  G2 MARGINAL → A3
  G2 PASS → A3
A3: multi-seed → G3 significance
  G3 FAIL → revert, accept 44.608
  G3 PARTIAL (Δ +0.5 to +1.0) → ship update, then A4 if 5% still target
  G3 PASS (Δ ≥ +1.0) → A4 if 5% still target, else ship update
A4: stack (SAM2 + LVLM) → per-lever +0.3 gates → final ship
```

Wall-clock budget cap: **6 weeks total** across A1+A2+A3+A4.
Exceed → ship whatever stack is alive at A3.

## Verification

1. **G1 recall report:** per-(seq, class) IoU≥0.5 recall vs NS predict.txt
   boxes + new-box count. Save to
   `diagnostics/results/grounding_dino/recall_vs_ns.md`.
2. **G2 HOTA report:** invoke standard TrackEval via `run_hota_eval.py`
   (or fork). Read `pedestrian_summary.txt` + `car_summary.txt` line 1
   col 0 = HOTA. Compute 3-seq pooled.
3. **G3 multi-seed paired-t:** reuse `run_paired_wilcoxon.py` format,
   compare 9 cells (3 seeds × 3 seqs) vs ship reference.
4. **Box-shape regression check:** mean/std of bbox area + aspect ratio
   per detector. Confirms no systematic geometric shift before passing G1.

## Final Write-up Trigger

Commit to writeup at whatever A3 or A4 delivers if:
- (G1 FAIL ∨ G2 FAIL ∨ G3 FAIL), OR
- Wall-clock exceeds 6 weeks, OR
- Multi-seed mean ≥ +2.23 (5% target achieved) → ship + paper update.

Whichever comes first.
