"""Per-row failure-class attribution.

Adds FN_ikun_coverage as class 0 (pre-empted by absence of iKUN cache row
at the GT-active frame). The remaining decision tree is per the design.
"""
from __future__ import annotations
import math
import pandas as pd


# Pre-registered thresholds (design §4).
ALIGNER_LOW_GMC = 0.3
IKUN_LOW_LOGIT  = 0.0


def _isnan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def attribute_row(row) -> str:
    gt           = row["gt_match"]
    pred         = row["pred_match"]
    frame_in_ikun= bool(row.get("ikun_frame_in_cache", 0))
    matched_tid  = row.get("matched_tracker_id", None)
    has_match    = matched_tid is not None and not _isnan(matched_tid)
    has_ikun     = not _isnan(row["ikun_logit"])
    det_hit      = row["detector_hit"]
    tr_assoc     = row["tracker_assoc"]
    gmc_score    = row["aligner_gmc_score"]
    ikun_log     = row["ikun_logit"]
    fuse_gate    = row["fusion_gate"]

    # Class 0: iKUN cascade never predicts on this frame at all.
    if gt == 1 and not frame_in_ikun:
        return "FN_ikun_coverage"
    # Class 1: detector miss — no detection at this frame for the class.
    if gt == 1 and not det_hit:
        return "FN_detector"
    # Class 2: detector fired but tracker failed to associate (no IoU match
    # to any tracker prediction, OR matched track flagged lost/switched).
    if gt == 1 and (not has_match or tr_assoc != "stable"):
        return "FN_tracker"
    # Class 3: aligner-side miss — both gmc and ikun signals weak.
    if gt == 1 and _isnan(gmc_score):
        return "FN_aligner"   # tracker matched but gmc cache absent
    if gt == 1 and gmc_score < ALIGNER_LOW_GMC and (
            _isnan(ikun_log) or ikun_log < IKUN_LOW_LOGIT):
        return "FN_aligner"
    # Class 4: fusion-gate veto despite at-least-one-signal-positive.
    if (gt == 1
            and (gmc_score >= ALIGNER_LOW_GMC
                 or (has_ikun and ikun_log >= IKUN_LOW_LOGIT))
            and (not _isnan(fuse_gate) and fuse_gate < 0)):
        return "FN_fusion"
    if gt == 1 and pred == 1:
        return "TP"
    if gt == 0 and pred == 1:
        return "FP"
    return "TN"


def attribute_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["failure_class"] = df.apply(attribute_row, axis=1)
    return df
