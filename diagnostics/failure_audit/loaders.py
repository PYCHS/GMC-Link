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
