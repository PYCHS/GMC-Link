"""Pre-T8 frame-coverage reconnaissance for the failure-mode audit.

The 4-stage attribution tree (det/track/aligner/fusion) assumes every GT
positive has an iKUN logit + tracker prediction at the same (frame, track_id).
Reality: the iKUN cascade cache subsamples 60 of ~371 frames per seq, and
some (seq, expr) cells have ZERO rows in the cascade dump. If GT-active
frames fall outside the iKUN sample, no fusion gate can fire — failure is
upstream of every recipe lever we tested.

This recon counts, per target cell:
  - n_gt_rows        : (frame, gt_track_id) pairs in the GT trajectory file
  - n_gt_frames      : distinct GT frames
  - n_ikun_frames    : distinct frames where iKUN cache contains the target expr
  - n_overlap_frames : intersection (no IoU yet — just frame-level)
  - n_tracker_frames : distinct frames in NeuralSORT predict.txt for class
  - n_gmc_frames     : distinct frames in GMC depth-aug aligner cache

Emits diagnostics/results/failure_audit/coverage_recon.md.
"""
from __future__ import annotations
from pathlib import Path
import json
import collections
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from diagnostics.failure_audit.loaders import (
    load_gt, load_ikun_logits, load_gmc_scores, load_tracker_assoc, _expr_class,
)

# Per project_phase5b memory the genuinely-unrecoverable cells are 0011 only.
# 0013 turning-cars/turning-vehicles GT directories do not exist; the earlier
# spec mis-listed them. Pedestrian-walking on 0011 in memory refers to
# `pedestrian-who-are-walking`, not the gendered `*-{men,women}` (those live
# on 0013 and are recoverable per phase5b).
CELLS = [
    ("turning-cars",                  "0011"),
    ("turning-vehicles",              "0011"),
    ("pedestrian-who-are-walking",    "0011"),
    # Sanity controls — recoverable per memory; expect overlap > 0:
    ("moving-cars",                   "0011"),
    ("parking-vehicles",              "0011"),
]


def recon_cell(expr: str, seq: str) -> dict:
    gt   = load_gt(REPO, seq, expr)
    ikun = load_ikun_logits(REPO, seq, expr)
    gmc  = load_gmc_scores(REPO, seq, expr)
    tr   = load_tracker_assoc(REPO, seq, expr)

    gt_frames   = set(gt["frame"].tolist())   if not gt.empty   else set()
    ikun_frames = set(ikun["frame"].tolist()) if not ikun.empty else set()
    gmc_frames  = set(gmc["frame"].tolist())  if not gmc.empty  else set()
    tr_frames   = set(tr["frame"].tolist())   if not tr.empty   else set()

    return {
        "expr": expr,
        "seq": seq,
        "cls": _expr_class(expr),
        "n_gt_rows":        len(gt),
        "n_gt_frames":      len(gt_frames),
        "gt_frame_range":   (min(gt_frames), max(gt_frames)) if gt_frames else None,
        "n_ikun_frames":    len(ikun_frames),
        "ikun_frame_range": (min(ikun_frames), max(ikun_frames)) if ikun_frames else None,
        "n_overlap_frames": len(gt_frames & ikun_frames),
        "n_tracker_frames": len(tr_frames),
        "n_gmc_frames":     len(gmc_frames),
        "coverage_pct":     (100.0 * len(gt_frames & ikun_frames) / len(gt_frames)
                             if gt_frames else 0.0),
    }


def main() -> None:
    out_dir = REPO / "diagnostics" / "results" / "failure_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Failure-Mode Audit — Coverage Reconnaissance",
          "",
          "Frame-level coverage check **before** the full per-row build_table.",
          "Tests whether each cell's GT-active frames fall inside the iKUN",
          "cascade cache window. Cells with `coverage_pct=0%` mean the",
          "cascade dump never predicts on the frames where the verb fires —",
          "failure is upstream of detector/tracker/aligner/fusion levers.",
          "",
          "| expr | seq | cls | n_gt_rows | gt_frames | gt_range | ikun_frames | ikun_range | overlap | coverage_pct |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    rows = [recon_cell(expr, seq) for expr, seq in CELLS]
    for r in rows:
        gt_r = f"{r['gt_frame_range'][0]}-{r['gt_frame_range'][1]}" if r['gt_frame_range'] else "—"
        ik_r = f"{r['ikun_frame_range'][0]}-{r['ikun_frame_range'][1]}" if r['ikun_frame_range'] else "—"
        md.append(f"| {r['expr']} | {r['seq']} | {r['cls']} | "
                  f"{r['n_gt_rows']} | {r['n_gt_frames']} | {gt_r} | "
                  f"{r['n_ikun_frames']} | {ik_r} | {r['n_overlap_frames']} | "
                  f"{r['coverage_pct']:.1f}% |")

    md += ["", "## Interpretation",
           "",
           "- `coverage_pct = 0%` → 100% of FN attributable to **FN_ikun_coverage**;",
           "  no recipe lever (det/track/aligner/fusion) can fire on these frames.",
           "- `coverage_pct > 0%` → proceed to per-row build_table for residual cells.",
           ""]

    (out_dir / "coverage_recon.md").write_text("\n".join(md))
    (out_dir / "coverage_recon.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
