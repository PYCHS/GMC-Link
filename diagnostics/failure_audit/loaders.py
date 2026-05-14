"""Source loaders for the failure-mode audit.

Each loader returns a tidy pandas DataFrame keyed on a subset of
(seq, frame, track_id, expr). Joins happen in build_table.py.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np


def load_gt(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Load GT track presence for (seq, expr) from refer-kitti/gt_template_old/."""
    gt_path = repo_root / "refer-kitti" / "gt_template_old" / seq / expr / "gt.txt"
    if not gt_path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "gt_match"])
    rows = []
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        rows.append((int(parts[0]), int(parts[1])))
    df = pd.DataFrame(rows, columns=["frame", "track_id"])
    df["gt_match"] = 1
    return df


def load_ikun_logits(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Flatten the nested iKUN cascade logit cache to (frame, track_id, ikun_logit).

    JSON layout: {seq: {frame: {track_id: {expr: [logit]}}}}.
    For pedestrian-walking-* expression family, returns rows where the expr
    name *startswith* the given family token (e.g. pedestrian-walking-women,
    pedestrian-walking-men); ikun_logit is the mean across matched exprs.
    """
    cache_path = repo_root / "iKUN" / "ikun_results_v1_cascade_full.json"
    cache = json.loads(cache_path.read_text())
    seq_data = cache.get(seq, {})
    rows = []
    for frame_str, track_dict in seq_data.items():
        frame = int(frame_str)
        for track_str, expr_dict in track_dict.items():
            track_id = int(track_str)
            matched = [v[0] for k, v in expr_dict.items() if _expr_match(k, expr)]
            if matched:
                rows.append((frame, track_id, float(np.mean(matched))))
    return pd.DataFrame(rows, columns=["frame", "track_id", "ikun_logit"])


def _expr_match(cache_expr: str, target: str) -> bool:
    """True if cache_expr matches the target family.

    Exact for non-family targets, prefix-match for pedestrian-walking-*.
    """
    if target == "pedestrian-walking":
        return cache_expr.startswith("pedestrian-walking") or \
               cache_expr in {"persons-who-are-walking", "people-who-are-walking",
                              "walking-pedestrian"}
    return cache_expr == target
