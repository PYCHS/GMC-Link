# Failure-Mode Audit: Cascade iKUN Unrecoverable Expressions

**Date:** 2026-05-14
**Branch:** `exp/ego-motion-systematic`
**Status:** Design — not yet implemented

## Motivation

Across 14+ aligner levers explored (Exp 34–43, depth-aug 17D, world-XY, CLIP fusion sites
1–4), three expressions remain unrecoverable on cascade iKUN even after Phase 5 conditional
recipes:

- `turning-cars` — universal unrecoverable across cascade + non-cascade arches
  (memory: `project_phase5b_per_expr_recovery_rate`, `project_phase5f_noncascade_per_expr`)
- `turning-vehicles` — same pattern
- `pedestrian-walking-{men,women}` on seq 0011 — gendered-token bias suspect; cascade and
  non-cascade arches exhibit *opposite* failure direction
  (memory: `project_phase5f_noncascade_per_expr`)

The HOTA pool ceiling on YOLOv8-NS is fixed at 44.564 (memory: `project_pooled_ceiling_44564`),
so further blind lever search is low-EV. A failure-stage breakdown will decide whether the
remaining headroom is detector-bound, tracker-bound, aligner-bound, fusion-gate-bound, or
GT/ambiguity-bound — pointing at the next experiment, or closing the door cleanly.

## Scope

- 1 arch: cascade iKUN (paper recipe, B = 44.224 pooled per
  `project_paper_repro_3seq_pooled`).
- 1 aligner: depth-aug 17D ship recipe (seed-1, single-seed reference per
  `project_ikun_multiseed_positive`).
- 5 (expression, sequence) cells:
  - `turning-cars` × 0011
  - `turning-cars` × 0013
  - `turning-vehicles` × 0011
  - `turning-vehicles` × 0013
  - `pedestrian-walking-*` × 0011
- GT convention pinned to `gt_template_old/` (paper-canonical, NeuralSORT-aligned per
  `project_gt_template_two_conventions`).

## Components

New directory `diagnostics/failure_audit/`.

### 1. `inventory.py`

Walk existing artifact paths (`det_cache/`, iKUN logit dumps under `iKUN/`, GMC score caches
under `diagnostics/results/depth_v1train/`, fusion-output cache from ship recipe). For each
of the 5 (expr, seq) cells, report which stages already have cached outputs and which
require a hook re-run. Output: `inventory.json`.

### 2. `hooks.py`

Lightweight logging shims inserted into `run_film_hota.py` (or its non-FiLM ship variant) at
each stage boundary. Each hook serializes per-frame, per-track tensors to a per-cell `.npz`
under `diagnostics/results/failure_audit/raw/`. Hooks only run for cells flagged by
`inventory.py`. No change to existing recipe outputs.

### 3. `build_table.py`

Join all sources on `(seq, frame, track_id, expr)`. iKUN logit frames are the canonical key
(left join). Unmatched rows are dropped and counted in `_run_log.txt`. Output:
`audit_<expr>_<seq>.parquet` with columns:

| column | source | type |
|---|---|---|
| `seq`, `frame`, `track_id`, `expr` | join key | str/int |
| `gt_match` | GT label join | 0/1 |
| `detector_hit` | det_cache | 0/1 |
| `tracker_assoc` | tracker state | {stable, switched, lost} |
| `aligner_gmc_score` | depth-aug ship cache | float ∈ [0,1] |
| `ikun_logit` | iKUN cascade dump | float |
| `fusion_gate` | ship-recipe output | float |
| `pred_match` | `fusion_gate ≥ 0` | 0/1 |

### 4. `attribute.py`

Per-row failure classification using ordered decision rule (first match wins):

1. `gt_match=1` AND `detector_hit=0` → `FN_detector`
2. `gt_match=1` AND `detector_hit=1` AND `tracker_assoc ∈ {switched, lost}` → `FN_tracker`
3. `gt_match=1` AND `tracker_assoc=stable` AND `aligner_gmc_score < 0.3` AND
   `ikun_logit < 0` → `FN_aligner`
4. `gt_match=1` AND `tracker_assoc=stable` AND at-least-one-signal-positive
   (`aligner_gmc_score ≥ 0.3` OR `ikun_logit ≥ 0`) AND `fusion_gate < 0`
   → `FN_fusion`
5. `gt_match=1` AND `pred_match=1` → `TP`
6. `gt_match=0` AND `pred_match=1` → `FP`
7. `gt_match=0` AND `pred_match=0` → `TN`

Thresholds `0.3` and `0` are pre-registered; chosen by inspection of ship-recipe distribution
(adjust once based on score histogram inspection during D1, then freeze).

### 5. `report.py`

Aggregate per (expr, seq) cell:

```
| stage         | n_FN | %_FN  | example_frames     |
|---------------|------|-------|--------------------|
| FN_detector   | ...  | ...   | seq0011-frame-... |
| FN_tracker    | ...  | ...   | ...                |
| FN_aligner    | ...  | ...   | ...                |
| FN_fusion     | ...  | ...   | ...                |
```

Plus pooled view across all 5 cells. Output: `SUMMARY.md`.

## Data Flow

```
det_cache/<seq>/predict.txt
  → (frame, track_id, bbox, det_score)
        │
        ▼
NeuralSORT tracker assoc state
  → tracker_assoc ∈ {stable, switched, lost}
        │
        ▼
iKUN cascade logits cache
  → ikun_logit
        │
        ▼
GMC depth-aug aligner cache (seed-1 ship)
  → aligner_gmc_score
        │
        ▼
Fusion gate (ship recipe: scale, threshold, α)
  → fusion_gate, pred_match
        │
        ▼
GT label join (gt_template_old)
  → gt_match
        │
        ▼
build_table.py → audit_<expr>_<seq>.parquet
        │
        ▼
attribute.py → adds failure_class column
        │
        ▼
report.py → SUMMARY.md
```

## Error Handling

- **Missing caches:** `inventory.py` flags, `hooks.py` re-runs only the missing recipe
  slice. No silent skip.
- **Frame-index mismatch between caches:** left-join on iKUN logit frames as canonical.
  Unmatched rows logged + dropped, counts surfaced in report. If >5% of frames drop,
  abort and investigate.
- **GT convention:** pin `gt_template_old/`; assert frame alignment on load.
- **Aligner seed:** single-seed seed-1 (typical, not cherry-picked per memory).
  Multi-seed mean would mask per-frame attribution signal.

## Testing

- **Unit:** `attribute.py` decision tree tested on synthetic rows covering each of the 7
  output classes.
- **Sanity:** total attributed (TP + FP + TN + sum(FN_*)) per (expr, seq) == total
  eval rows for that cell.
- **Cross-check:** TP frame count per (expr, seq) reproduces that cell's HOTA-DetA
  contribution within ±0.5 vs cascade B HOTA on `gt_template_old` (44.224 pooled).
  Larger drift indicates join error.

## Deliverables

- `diagnostics/results/failure_audit/raw/<expr>_<seq>.npz` — per-stage tensors
- `diagnostics/results/failure_audit/audit_<expr>_<seq>.parquet` — joined table
- `diagnostics/results/failure_audit/SUMMARY.md` — per-cell + pooled breakdown
- `diagnostics/results/failure_audit/_run_log.txt` — recipes, seeds, cache provenance

## Decision Criteria

After report assembly:

- **One stage ≥ 60% of FN** across all 5 cells → "lever found"; spec next experiment
  targeting that stage.
- **No stage ≥ 60% but one stage ≥ 40%** → mixed; spec experiment for worst-stage cells
  only.
- **All stages < 40% AND high TN rate on motion frames** → GT noise / motion ambiguity;
  door closes, ship current results, move to write-up.

## Timeline

~3 days:

- D1: inventory + hooks (plumb into existing run scripts; verify caches dump correctly
  on one cell).
- D2: build_table + attribute + tests.
- D3: report + decision write-up.

## Non-Goals

- No new training. No new aligner. No new recipe sweep.
- Does not target FH V1 / FH V2 archs. Cascade iKUN only.
- Does not aim to close the 44.224 → 44.564 → 48.84 SOTA gap; aims only to identify the
  dominant failure stage for the 3 known unrecoverable expression families.
