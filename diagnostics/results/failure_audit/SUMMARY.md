# Failure-Mode Audit — Pre-T8 Reconnaissance Verdict

**Date:** 2026-05-14
**Branch:** `exp/ego-motion-systematic`
**Scope:** 3 cells flagged "unrecoverable" by `project_phase5b_per_expr_recovery_rate`

## TL;DR

The pre-T8 coverage recon proved that **two of three** "unrecoverable" cells
on cascade iKUN are not bound by detector/tracker/aligner/fusion — they are
upstream-bound by **iKUN's V1 test-split frame window** (frames 1-114 on
seq 0011) which has **zero temporal overlap** with the GT-active range
(frames 309-338).

The remaining cell (`pedestrian-who-are-walking` × 0011) has 44.6% frame
overlap and is the only candidate for the planned 4-stage attribution.

## Coverage table

| expr | seq | n_gt_rows | GT frame range | iKUN frame range | overlap | coverage |
|---|---|---:|---|---|---:|---:|
| `turning-cars`               | 0011 |   30 | 309-338 | 1-114 |  0 |  **0.0 %** |
| `turning-vehicles`           | 0011 |   30 | 309-338 | 1-114 |  0 |  **0.0 %** |
| `pedestrian-who-are-walking` | 0011 |   90 |   1-56  | 1-114 | 25 |   44.6 %  |
| `moving-cars` (control)      | 0011 |  650 |   1-371 | 1-114 | 60 |   19.0 %  |
| `parking-vehicles` (control) | 0011 | 2598 |   1-355 | 1-114 | 60 |   16.9 %  |

Controls (`moving-cars`, `parking-vehicles`) score ≥30 HOTA at 17-19 %
coverage — partial sampling is *sufficient* when GT and iKUN windows
overlap. The two turning-verb cells fail because their GT temporal extent
sits **entirely outside** iKUN's evaluation window on seq 0011.

## Failure-class attribution

| cell | dominant class | actionable lever |
|---|---|---|
| `turning-cars` × 0011               | **FN_ikun_coverage** (100 %) | Expand iKUN V1 frame sampling beyond frame 114 on 0011; no recipe-side fix possible. |
| `turning-vehicles` × 0011           | **FN_ikun_coverage** (100 %) | Same as above. |
| `pedestrian-who-are-walking` × 0011 | mixed — proceed to per-row T8  | Full det/track/aligner/fusion attribution still warranted. |

## Correction to phase5b memory

`project_phase5b_per_expr_recovery_rate` claims:

> "Cascade logits all below -0.5 for turning exprs → no admit at any threshold."

Recon shows this is incorrect. There are **no cascade logits** at GT-active
frames for `turning-cars`/`turning-vehicles` on 0011, not low-magnitude
logits. The iKUN cache simply does not predict on frames 309-338.

## Decision

Per the design doc's decision criteria:

> "All stages < 40 % AND high TN rate on motion frames → GT noise / motion
> ambiguity; door closes, ship current results, move to write-up."

A fifth, unforeseen class — **FN_ikun_coverage** — dominates 2 of 3 cells.
This is not GT noise. It is an **upstream test-split sampling limitation**
that **no aligner / fusion lever can address**. Door closes for the two
turning-verb cells.

Remaining work: per-row T8 attribution for `pedestrian-who-are-walking`
× 0011 only. If that cell also attributes to a single dominant stage,
the audit yields a sharper next-experiment recommendation.

## Artifacts

- `diagnostics/results/failure_audit/coverage_recon.md` — this table
- `diagnostics/results/failure_audit/coverage_recon.json` — raw counts
- `diagnostics/failure_audit/coverage_recon.py` — script
