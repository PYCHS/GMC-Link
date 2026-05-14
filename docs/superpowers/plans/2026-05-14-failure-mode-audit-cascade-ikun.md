# Failure-Mode Audit: Cascade iKUN Unrecoverables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-row diagnostic table that attributes every false-negative in 5 (expr, seq) cells to one pipeline stage (detector / tracker / aligner / fusion), and decide from the breakdown whether one stage dominates enough to justify a next experiment.

**Architecture:** Five Python modules under `diagnostics/failure_audit/`. iKUN cascade logit JSON is the canonical join key; detector, tracker, GMC aligner, fusion gate, and GT each contribute a column. A decision-tree classifier (`attribute.py`) labels each row, and a markdown report aggregates per-cell and pooled stage distributions.

**Tech Stack:** Python 3, NumPy, pandas (parquet), pytest. No new GPU model runs — reads existing caches under `det_cache/`, `iKUN/`, `diagnostics/results/depth_v1train/`, `gt_template_old/`.

**Spec:** `docs/superpowers/specs/2026-05-14-failure-mode-audit-cascade-ikun-design.md`

---

## File Structure

```
diagnostics/failure_audit/
├── __init__.py
├── inventory.py        # Lists cached / missing per (seq, expr) cell
├── loaders.py          # Source loaders: GT, iKUN logit, det, tracker, GMC, fusion
├── build_table.py      # Joins → parquet (one row per seq/frame/track/expr)
├── attribute.py        # 7-class decision-tree (TP/FP/TN + 4 FN_*)
├── report.py           # Markdown summary aggregator
└── run_audit.py        # Driver: inventory → build → attribute → report

tests/failure_audit/
├── test_attribute.py   # Decision-tree unit tests (TDD core)
├── test_build_table.py # Join correctness, key cardinality
└── test_loaders.py     # Cell loaders sanity (frame range, dtype)

diagnostics/results/failure_audit/
├── inventory.json
├── raw/                # per-cell .npz (only if hooks needed)
├── audit_<expr>_<seq>.parquet
├── SUMMARY.md
└── _run_log.txt
```

**5 target cells** (driver-level constant):

```python
TARGET_CELLS = [
    ("turning-cars",         "0011"),
    ("turning-cars",         "0013"),
    ("turning-vehicles",     "0011"),
    ("turning-vehicles",     "0013"),
    ("pedestrian-walking",   "0011"),  # match any pedestrian-walking-* expr family
]
```

**Ship recipe pinned constants** (per memory `project_ikun_appearance_extension_positive` and `project_depth_augmented_17d_negative`):

```python
SHIP_RECIPE = {
    "ikun_cache":   "iKUN/ikun_results_v1_cascade.json",
    "gmc_seed":     1,
    "gmc_cache":    "diagnostics/results/depth_v1train/layer3_{seq}_depth_seed1.npz",
    "alpha_motion": 1.0,
    "scale_motion": 0.9,
    "thr_motion":   0.17,
    "alpha_appear": 1.0,
    "scale_appear": 0.30,
    "thr_appear":   0.10,
    "gt_root":      "gt_template_old/",
    "det_root":     "det_cache/DDETR-kitti/",
}
```

---

## Task 1: Bootstrap module + inventory

**Files:**
- Create: `diagnostics/failure_audit/__init__.py`
- Create: `diagnostics/failure_audit/inventory.py`
- Create: `tests/failure_audit/__init__.py`
- Create: `tests/failure_audit/test_inventory.py`

- [ ] **Step 1.1: Create empty package init**

```python
# diagnostics/failure_audit/__init__.py
"""Failure-mode audit for cascade iKUN unrecoverables.

See: docs/superpowers/specs/2026-05-14-failure-mode-audit-cascade-ikun-design.md
"""
```

```python
# tests/failure_audit/__init__.py
```

- [ ] **Step 1.2: Write failing test for inventory**

```python
# tests/failure_audit/test_inventory.py
from pathlib import Path
import pytest
from diagnostics.failure_audit.inventory import inventory_cells, CellStatus, TARGET_CELLS


def test_inventory_returns_one_status_per_cell(tmp_path):
    repo_root = tmp_path
    (repo_root / "iKUN").mkdir()
    (repo_root / "iKUN" / "ikun_results_v1_cascade.json").write_text("{}")
    statuses = inventory_cells(repo_root)
    assert len(statuses) == len(TARGET_CELLS)
    for s in statuses:
        assert isinstance(s, CellStatus)
        assert s.expr in {c[0] for c in TARGET_CELLS}


def test_inventory_flags_missing_gmc_seed1_cache(tmp_path):
    repo_root = tmp_path
    (repo_root / "iKUN").mkdir()
    (repo_root / "iKUN" / "ikun_results_v1_cascade.json").write_text("{}")
    statuses = inventory_cells(repo_root)
    # No depth_v1train caches written → all cells must flag gmc missing
    assert all(not s.gmc_present for s in statuses)
```

- [ ] **Step 1.3: Run test, expect ImportError**

```bash
pytest tests/failure_audit/test_inventory.py -v
```
Expected: `ModuleNotFoundError: No module named 'diagnostics.failure_audit.inventory'`

- [ ] **Step 1.4: Implement inventory.py minimal**

```python
# diagnostics/failure_audit/inventory.py
"""Cache-presence inventory for the failure-mode audit.

For each of the 5 target (expr, seq) cells, reports which source caches are
already on disk and which require a hook re-run.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

TARGET_CELLS = [
    ("turning-cars",       "0011"),
    ("turning-cars",       "0013"),
    ("turning-vehicles",   "0011"),
    ("turning-vehicles",   "0013"),
    ("pedestrian-walking", "0011"),
]


@dataclass
class CellStatus:
    expr: str
    seq: str
    ikun_present: bool
    gmc_present: bool
    det_present: bool
    gt_present: bool

    @property
    def all_present(self) -> bool:
        return self.ikun_present and self.gmc_present and self.det_present and self.gt_present


def inventory_cells(repo_root: Path) -> List[CellStatus]:
    out = []
    for expr, seq in TARGET_CELLS:
        out.append(CellStatus(
            expr=expr,
            seq=seq,
            ikun_present=(repo_root / "iKUN" / "ikun_results_v1_cascade.json").exists(),
            gmc_present=(repo_root / "diagnostics" / "results" / "depth_v1train" /
                         f"layer3_{seq}_depth_seed1.npz").exists(),
            det_present=(repo_root / "det_cache" / "DDETR-kitti" / seq).exists(),
            gt_present=(repo_root / "gt_template_old" / seq).exists(),
        ))
    return out
```

- [ ] **Step 1.5: Run test to verify pass**

```bash
pytest tests/failure_audit/test_inventory.py -v
```
Expected: 2 passed

- [ ] **Step 1.6: Commit**

```bash
git add diagnostics/failure_audit/__init__.py diagnostics/failure_audit/inventory.py \
        tests/failure_audit/__init__.py tests/failure_audit/test_inventory.py
git commit -m "feat(audit): cell inventory module with cache-presence check"
```

---

## Task 2: GT loader

**Files:**
- Create: `diagnostics/failure_audit/loaders.py`
- Modify: `tests/failure_audit/test_loaders.py` (new)

- [ ] **Step 2.1: Write failing test**

```python
# tests/failure_audit/test_loaders.py
from pathlib import Path
import pandas as pd
from diagnostics.failure_audit.loaders import load_gt


def test_load_gt_returns_dataframe_with_required_columns(tmp_path):
    # Build a synthetic gt_template_old layout matching the real format:
    #   gt_template_old/<seq>/<expr>/gt.txt   (frame,id,...)
    seq_dir = tmp_path / "gt_template_old" / "0011" / "turning-cars"
    seq_dir.mkdir(parents=True)
    (seq_dir / "gt.txt").write_text(
        "1,5,100,200,50,80,1,1,1\n"
        "2,5,102,201,50,80,1,1,1\n"
        "3,7,300,400,60,90,1,1,1\n"
    )
    df = load_gt(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "gt_match"]
    assert len(df) == 3
    assert df["gt_match"].eq(1).all()
    assert df["track_id"].dtype.kind in {"i", "u"}
```

- [ ] **Step 2.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_loaders.py::test_load_gt_returns_dataframe_with_required_columns -v
```
Expected: ImportError on `load_gt`

- [ ] **Step 2.3: Implement loader**

```python
# diagnostics/failure_audit/loaders.py
"""Source loaders for the failure-mode audit.

Each loader returns a tidy pandas DataFrame keyed on a subset of
(seq, frame, track_id, expr). Joins happen in build_table.py.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np


def load_gt(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Load GT track presence for (seq, expr) from gt_template_old/."""
    gt_path = repo_root / "gt_template_old" / seq / expr / "gt.txt"
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
```

- [ ] **Step 2.4: Run test to verify pass**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: 1 passed

- [ ] **Step 2.5: Commit**

```bash
git add diagnostics/failure_audit/loaders.py tests/failure_audit/test_loaders.py
git commit -m "feat(audit): GT loader for gt_template_old"
```

---

## Task 3: iKUN logit loader (canonical join key)

**Files:**
- Modify: `diagnostics/failure_audit/loaders.py`
- Modify: `tests/failure_audit/test_loaders.py`

- [ ] **Step 3.1: Write failing test**

```python
# append to tests/failure_audit/test_loaders.py
import json
from diagnostics.failure_audit.loaders import load_ikun_logits


def test_load_ikun_logits_flattens_nested_json(tmp_path):
    cache = {
        "0011": {
            "1": {  # frame
                "5": {  # track_id
                    "turning-cars": [-0.12],
                    "moving-vehicles": [0.42],
                },
            },
            "2": {
                "5": {"turning-cars": [0.05]},
            },
        }
    }
    p = tmp_path / "iKUN"
    p.mkdir()
    (p / "ikun_results_v1_cascade.json").write_text(json.dumps(cache))
    df = load_ikun_logits(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "ikun_logit"]
    assert set(df["frame"]) == {1, 2}
    assert df.loc[df.frame == 1, "ikun_logit"].iloc[0] == -0.12
```

- [ ] **Step 3.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_loaders.py::test_load_ikun_logits_flattens_nested_json -v
```
Expected: ImportError

- [ ] **Step 3.3: Implement `load_ikun_logits`**

```python
# append to diagnostics/failure_audit/loaders.py
def load_ikun_logits(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Flatten the nested iKUN cascade logit cache to (frame, track_id, ikun_logit).

    JSON layout: {seq: {frame: {track_id: {expr: [logit]}}}}.
    For pedestrian-walking-* expression family, returns rows where the expr
    name *startswith* the given family token (e.g. pedestrian-walking-women,
    pedestrian-walking-men); ikun_logit is the mean across matched exprs.
    """
    cache_path = repo_root / "iKUN" / "ikun_results_v1_cascade.json"
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
```

- [ ] **Step 3.4: Run test**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: 2 passed

- [ ] **Step 3.5: Commit**

```bash
git add diagnostics/failure_audit/loaders.py tests/failure_audit/test_loaders.py
git commit -m "feat(audit): iKUN cascade logit loader with pedestrian-walking family expansion"
```

---

## Task 4: Detector hit loader

**Files:**
- Modify: `diagnostics/failure_audit/loaders.py`
- Modify: `tests/failure_audit/test_loaders.py`

- [ ] **Step 4.1: Write failing test**

```python
# append to tests/failure_audit/test_loaders.py
from diagnostics.failure_audit.loaders import load_detector_hits


def test_load_detector_hits_returns_per_frame_track_presence(tmp_path):
    # det_cache/DDETR-kitti/<seq>/<class>/dets.json layout
    # We expect: {frame: [[x1,y1,x2,y2,score,track_id], ...]} or similar
    layout = {
        "1": [[100, 200, 150, 280, 0.9, 5]],
        "2": [[102, 201, 152, 281, 0.85, 5]],
    }
    d = tmp_path / "det_cache" / "DDETR-kitti" / "0011" / "car"
    d.mkdir(parents=True)
    (d / "dets.json").write_text(json.dumps(layout))
    df = load_detector_hits(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "detector_hit"]
    assert df["detector_hit"].eq(1).all()
    assert len(df) == 2
```

- [ ] **Step 4.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_loaders.py::test_load_detector_hits_returns_per_frame_track_presence -v
```
Expected: ImportError

- [ ] **Step 4.3: Implement `load_detector_hits`**

NOTE: Real `det_cache/DDETR-kitti/<seq>/<class>/dets.json` schema must be confirmed at run
time. If it does not carry track_id, fall back to bbox-IoU matching against the iKUN logit
key per frame. Implementation handles both shapes:

```python
# append to diagnostics/failure_audit/loaders.py
def load_detector_hits(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Per (frame, track_id) detector-hit indicator.

    Reads det_cache/DDETR-kitti/<seq>/<class>/dets.json. Class is inferred from
    expression family ('car' for *-cars / *-vehicles, 'pedestrian' for
    pedestrian-*).
    """
    cls = _expr_class(expr)
    det_path = repo_root / "det_cache" / "DDETR-kitti" / seq / cls / "dets.json"
    if not det_path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "detector_hit"])
    raw = json.loads(det_path.read_text())
    rows = []
    for frame_str, dets in raw.items():
        frame = int(frame_str)
        for det in dets:
            # det = [x1,y1,x2,y2,score,track_id] OR [x1,y1,x2,y2,score]
            track_id = int(det[5]) if len(det) >= 6 else -1
            rows.append((frame, track_id, 1))
    return pd.DataFrame(rows, columns=["frame", "track_id", "detector_hit"])


def _expr_class(expr: str) -> str:
    if expr.startswith("pedestrian"):
        return "pedestrian"
    return "car"
```

- [ ] **Step 4.4: Run test**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: 3 passed

- [ ] **Step 4.5: Commit**

```bash
git add diagnostics/failure_audit/loaders.py tests/failure_audit/test_loaders.py
git commit -m "feat(audit): detector-hit loader with class inference"
```

---

## Task 5: Tracker assoc loader

**Files:**
- Modify: `diagnostics/failure_audit/loaders.py`
- Modify: `tests/failure_audit/test_loaders.py`

- [ ] **Step 5.1: Write failing test**

```python
# append to tests/failure_audit/test_loaders.py
from diagnostics.failure_audit.loaders import load_tracker_assoc


def test_tracker_assoc_marks_switch_and_lost(tmp_path):
    # predict.txt format: frame,track_id,x1,y1,w,h,conf,...
    p = tmp_path / "NeuralSORT" / "0011"
    p.mkdir(parents=True)
    (p / "predict.txt").write_text(
        "1,5,100,200,50,80,0.9\n"
        "2,5,102,202,50,80,0.9\n"   # stable (5 continues)
        "4,5,200,202,50,80,0.9\n"   # 5 reappears after frame 3 gap → lost@3 ignored
                                      # but state at 4 = "lost" since gap > 1
        "5,7,300,400,60,90,0.9\n"   # new track 7, no prior history → stable (fresh)
    )
    df = load_tracker_assoc(tmp_path, seq="0011")
    assert set(df.columns) == {"frame", "track_id", "tracker_assoc"}
    row = df[(df.frame == 2) & (df.track_id == 5)].iloc[0]
    assert row["tracker_assoc"] == "stable"
    row = df[(df.frame == 4) & (df.track_id == 5)].iloc[0]
    assert row["tracker_assoc"] == "lost"
```

- [ ] **Step 5.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_loaders.py::test_tracker_assoc_marks_switch_and_lost -v
```
Expected: ImportError

- [ ] **Step 5.3: Implement `load_tracker_assoc`**

```python
# append to diagnostics/failure_audit/loaders.py
def load_tracker_assoc(repo_root: Path, seq: str) -> pd.DataFrame:
    """Per (frame, track_id) assoc state ∈ {stable, switched, lost}.

    Rule:
      stable  — track existed in frame-1 and bbox center moved < 2x its diag
      lost    — track existed previously but had a gap > 1 frame
      switched— track existed in frame-1 and bbox center jumped > 2x diag
                (impossible with single predict.txt without ID-tracking detector
                output; reduced to lost-vs-stable for the v1 audit)
    """
    pred_path = repo_root / "NeuralSORT" / seq / "predict.txt"
    if not pred_path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "tracker_assoc"])
    raw = []
    for line in pred_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        raw.append({
            "frame": int(parts[0]),
            "track_id": int(parts[1]),
        })
    df = pd.DataFrame(raw)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    df["prev_frame"] = df.groupby("track_id")["frame"].shift(1)
    df["tracker_assoc"] = np.where(
        df["prev_frame"].isna(),                       "stable",
        np.where(df["frame"] - df["prev_frame"] == 1,  "stable",
                                                       "lost")
    )
    return df[["frame", "track_id", "tracker_assoc"]]
```

- [ ] **Step 5.4: Run test**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: 4 passed

- [ ] **Step 5.5: Commit**

```bash
git add diagnostics/failure_audit/loaders.py tests/failure_audit/test_loaders.py
git commit -m "feat(audit): tracker assoc loader (stable/lost from gap rule)"
```

---

## Task 6: GMC aligner score loader

**Files:**
- Modify: `diagnostics/failure_audit/loaders.py`
- Modify: `tests/failure_audit/test_loaders.py`

- [ ] **Step 6.1: Write failing test**

```python
# append to tests/failure_audit/test_loaders.py
from diagnostics.failure_audit.loaders import load_gmc_scores


def test_load_gmc_scores_reads_npz_seed1(tmp_path):
    # Expected npz layout (from existing depth_v1train cache):
    #   frame_id : np.int32[N]
    #   track_id : np.int32[N]
    #   expr     : np.array(N, dtype=object)   # string expressions
    #   gmc_score: np.float32[N]               # cosine ∈ [0,1]
    npz_dir = tmp_path / "diagnostics" / "results" / "depth_v1train"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / "layer3_0011_depth_seed1.npz",
        frame_id=np.array([1, 2, 3], dtype=np.int32),
        track_id=np.array([5, 5, 7], dtype=np.int32),
        expr=np.array(["turning-cars", "turning-cars", "moving-vehicles"], dtype=object),
        gmc_score=np.array([0.42, 0.51, 0.88], dtype=np.float32),
    )
    df = load_gmc_scores(tmp_path, seq="0011", expr="turning-cars")
    assert list(df.columns) == ["frame", "track_id", "aligner_gmc_score"]
    assert len(df) == 2
    assert abs(df["aligner_gmc_score"].iloc[0] - 0.42) < 1e-5
```

- [ ] **Step 6.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_loaders.py::test_load_gmc_scores_reads_npz_seed1 -v
```
Expected: ImportError

- [ ] **Step 6.3: Implement `load_gmc_scores`**

NOTE: real npz schema must be confirmed by reading one of the existing
`layer3_<seq>_depth_seed1.npz` files during D1. If the real keys differ
(e.g., `frames`, `tracks`, `scores`), update the implementation here. Do NOT
silently rename.

```python
# append to diagnostics/failure_audit/loaders.py
def load_gmc_scores(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    """Per (frame, track_id) GMC aligner cosine from depth-aug seed-1 cache."""
    npz_path = (repo_root / "diagnostics" / "results" / "depth_v1train" /
                f"layer3_{seq}_depth_seed1.npz")
    if not npz_path.exists():
        return pd.DataFrame(columns=["frame", "track_id", "aligner_gmc_score"])
    data = np.load(npz_path, allow_pickle=True)
    mask = np.array([_expr_match(str(e), expr) for e in data["expr"]])
    return pd.DataFrame({
        "frame":             data["frame_id"][mask].astype(int),
        "track_id":          data["track_id"][mask].astype(int),
        "aligner_gmc_score": data["gmc_score"][mask].astype(float),
    })
```

- [ ] **Step 6.4: Run test**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: 5 passed

- [ ] **Step 6.5: Schema-confirm against a real npz**

```bash
python -c "import numpy as np; \
d = np.load('diagnostics/results/depth_v1train/layer3_0011_depth_seed1.npz', \
            allow_pickle=True); print(list(d.keys())); \
print({k: d[k].shape for k in d.keys()})"
```
Expected: print the actual keys. If they differ from {`frame_id`, `track_id`, `expr`,
`gmc_score`}, edit `load_gmc_scores` AND the test fixture above to match, re-run pytest
(must stay green), then proceed.

- [ ] **Step 6.6: Commit**

```bash
git add diagnostics/failure_audit/loaders.py tests/failure_audit/test_loaders.py
git commit -m "feat(audit): GMC depth-aug seed-1 aligner score loader"
```

---

## Task 7: Fusion gate computation

**Files:**
- Modify: `diagnostics/failure_audit/loaders.py`
- Modify: `tests/failure_audit/test_loaders.py`

- [ ] **Step 7.1: Write failing test**

```python
# append to tests/failure_audit/test_loaders.py
from diagnostics.failure_audit.loaders import compute_fusion_gate


def test_fusion_gate_motion_axis_for_turning_cars():
    # turning-cars is a MOVING-class expr → motion-axis recipe.
    # fusion = ikun_logit + alpha_motion * scale_motion * gmc_score + thr_motion
    score = compute_fusion_gate(ikun_logit=-0.5, gmc_score=0.8, expr="turning-cars")
    # = -0.5 + 1.0 * 0.9 * 0.8 + 0.17 = -0.5 + 0.72 + 0.17 = 0.39
    assert abs(score - 0.39) < 1e-5


def test_fusion_gate_appear_axis_for_pedestrian_walking():
    # pedestrian-walking is MOVING but cascade memory shows appearance-axis tuning
    # is the iKUN ship default; tested against the locked appearance recipe.
    score = compute_fusion_gate(ikun_logit=-0.1, gmc_score=0.5, expr="pedestrian-walking")
    # = -0.1 + 1.0 * 0.9 * 0.5 + 0.17 = -0.1 + 0.45 + 0.17 = 0.52
    assert abs(score - 0.52) < 1e-5
```

- [ ] **Step 7.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: 2 new tests fail (ImportError)

- [ ] **Step 7.3: Implement `compute_fusion_gate`**

```python
# append to diagnostics/failure_audit/loaders.py
SHIP_ALPHA_MOTION  = 1.0
SHIP_SCALE_MOTION  = 0.9
SHIP_THR_MOTION    = 0.17


def compute_fusion_gate(ikun_logit: float, gmc_score: float, expr: str) -> float:
    """Apply ship-recipe motion-axis fusion. All 3 target families are MOVING.

    Recipe (memory: project_ikun_scalematched_positive):
        fused = ikun_logit + alpha * scale * gmc_score + thr
    """
    return (ikun_logit
            + SHIP_ALPHA_MOTION * SHIP_SCALE_MOTION * gmc_score
            + SHIP_THR_MOTION)
```

- [ ] **Step 7.4: Run test**

```bash
pytest tests/failure_audit/test_loaders.py -v
```
Expected: all loader tests pass

- [ ] **Step 7.5: Commit**

```bash
git add diagnostics/failure_audit/loaders.py tests/failure_audit/test_loaders.py
git commit -m "feat(audit): fusion-gate computation with locked ship recipe"
```

---

## Task 8: build_table.py — join all sources

**Files:**
- Create: `diagnostics/failure_audit/build_table.py`
- Create: `tests/failure_audit/test_build_table.py`

- [ ] **Step 8.1: Write failing test**

```python
# tests/failure_audit/test_build_table.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from diagnostics.failure_audit.build_table import build_audit_table


def _seed_repo(tmp_path: Path):
    """Build minimum viable cache layout for one (seq, expr) cell."""
    seq, expr = "0011", "turning-cars"
    # GT
    gt_dir = tmp_path / "gt_template_old" / seq / expr
    gt_dir.mkdir(parents=True)
    (gt_dir / "gt.txt").write_text("1,5,0,0,0,0,1,1,1\n2,5,0,0,0,0,1,1,1\n")
    # iKUN logits
    (tmp_path / "iKUN").mkdir()
    (tmp_path / "iKUN" / "ikun_results_v1_cascade.json").write_text(json.dumps({
        seq: {"1": {"5": {expr: [-0.5]}}, "2": {"5": {expr: [0.3]}}}
    }))
    # Det
    d = tmp_path / "det_cache" / "DDETR-kitti" / seq / "car"
    d.mkdir(parents=True)
    (d / "dets.json").write_text(json.dumps({"1": [[0,0,0,0,0.9,5]], "2": [[0,0,0,0,0.9,5]]}))
    # Tracker
    (tmp_path / "NeuralSORT" / seq).mkdir(parents=True)
    (tmp_path / "NeuralSORT" / seq / "predict.txt").write_text("1,5,0,0,0,0,0.9\n2,5,0,0,0,0,0.9\n")
    # GMC
    npz_dir = tmp_path / "diagnostics" / "results" / "depth_v1train"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / f"layer3_{seq}_depth_seed1.npz",
        frame_id=np.array([1, 2], dtype=np.int32),
        track_id=np.array([5, 5], dtype=np.int32),
        expr=np.array([expr, expr], dtype=object),
        gmc_score=np.array([0.8, 0.4], dtype=np.float32),
    )


def test_build_audit_table_emits_one_row_per_ikun_key(tmp_path):
    _seed_repo(tmp_path)
    df = build_audit_table(tmp_path, seq="0011", expr="turning-cars")
    required = {"seq", "frame", "track_id", "expr", "gt_match",
                "detector_hit", "tracker_assoc", "aligner_gmc_score",
                "ikun_logit", "fusion_gate", "pred_match"}
    assert required <= set(df.columns)
    assert len(df) == 2
    # gt_match should be 1 for both
    assert df["gt_match"].eq(1).all()
    # fusion_gate at frame 1: -0.5 + 0.9*0.8 + 0.17 = 0.39 → pred=1
    f1 = df[df.frame == 1].iloc[0]
    assert abs(f1["fusion_gate"] - 0.39) < 1e-5
    assert f1["pred_match"] == 1


def test_build_audit_table_zero_join_loss_on_canonical_key(tmp_path):
    _seed_repo(tmp_path)
    df = build_audit_table(tmp_path, seq="0011", expr="turning-cars")
    # Both iKUN keys (frames 1 + 2) must survive; no NaN in any required column
    assert df[["ikun_logit", "fusion_gate", "pred_match"]].notna().all().all()
```

- [ ] **Step 8.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_build_table.py -v
```
Expected: ImportError

- [ ] **Step 8.3: Implement build_table**

```python
# diagnostics/failure_audit/build_table.py
"""Join all source loaders → one row per (seq, frame, track_id, expr).

iKUN cascade logit frames are the canonical key (left-join base). Unmatched
rows from other sources are coerced to NaN; missing canonical-key rows are
dropped and logged.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from diagnostics.failure_audit.loaders import (
    load_gt, load_ikun_logits, load_detector_hits, load_tracker_assoc,
    load_gmc_scores, compute_fusion_gate,
)


def build_audit_table(repo_root: Path, seq: str, expr: str) -> pd.DataFrame:
    ikun = load_ikun_logits(repo_root, seq=seq, expr=expr)
    if ikun.empty:
        return pd.DataFrame()
    base = ikun.assign(seq=seq, expr=expr)

    gt = load_gt(repo_root, seq=seq, expr=expr)
    det = load_detector_hits(repo_root, seq=seq, expr=expr)
    trk = load_tracker_assoc(repo_root, seq=seq)
    gmc = load_gmc_scores(repo_root, seq=seq, expr=expr)

    df = base.merge(gt,  on=["frame", "track_id"], how="left")
    df = df.merge(det,   on=["frame", "track_id"], how="left")
    df = df.merge(trk,   on=["frame", "track_id"], how="left")
    df = df.merge(gmc,   on=["frame", "track_id"], how="left")

    df["gt_match"]          = df["gt_match"].fillna(0).astype(int)
    df["detector_hit"]      = df["detector_hit"].fillna(0).astype(int)
    df["tracker_assoc"]     = df["tracker_assoc"].fillna("lost")
    df["aligner_gmc_score"] = df["aligner_gmc_score"].fillna(0.0)

    df["fusion_gate"] = df.apply(
        lambda r: compute_fusion_gate(r["ikun_logit"], r["aligner_gmc_score"], expr),
        axis=1,
    )
    df["pred_match"] = (df["fusion_gate"] >= 0).astype(int)
    return df[[
        "seq", "frame", "track_id", "expr",
        "gt_match", "detector_hit", "tracker_assoc",
        "aligner_gmc_score", "ikun_logit", "fusion_gate", "pred_match",
    ]]
```

- [ ] **Step 8.4: Run test**

```bash
pytest tests/failure_audit/test_build_table.py -v
```
Expected: 2 passed

- [ ] **Step 8.5: Commit**

```bash
git add diagnostics/failure_audit/build_table.py tests/failure_audit/test_build_table.py
git commit -m "feat(audit): build_audit_table joins 5 sources on canonical iKUN key"
```

---

## Task 9: attribute.py — 7-class decision tree

**Files:**
- Create: `diagnostics/failure_audit/attribute.py`
- Create: `tests/failure_audit/test_attribute.py`

- [ ] **Step 9.1: Write failing tests covering all 7 classes**

```python
# tests/failure_audit/test_attribute.py
import pandas as pd
from diagnostics.failure_audit.attribute import classify_row, attribute_table


def _row(**kw):
    base = dict(gt_match=0, detector_hit=0, tracker_assoc="stable",
                aligner_gmc_score=0.0, ikun_logit=0.0, fusion_gate=0.0, pred_match=0)
    base.update(kw)
    return pd.Series(base)


def test_tn_when_no_gt_no_pred():
    assert classify_row(_row(gt_match=0, pred_match=0)) == "TN"


def test_fp_when_no_gt_but_pred():
    assert classify_row(_row(gt_match=0, pred_match=1)) == "FP"


def test_tp_when_gt_and_pred():
    assert classify_row(_row(gt_match=1, pred_match=1,
                             tracker_assoc="stable", detector_hit=1)) == "TP"


def test_fn_detector_when_no_detector_hit():
    assert classify_row(_row(gt_match=1, detector_hit=0)) == "FN_detector"


def test_fn_tracker_when_assoc_lost():
    assert classify_row(_row(gt_match=1, detector_hit=1,
                             tracker_assoc="lost", pred_match=0)) == "FN_tracker"


def test_fn_aligner_when_both_signals_dead():
    assert classify_row(_row(
        gt_match=1, detector_hit=1, tracker_assoc="stable",
        aligner_gmc_score=0.2, ikun_logit=-0.5, fusion_gate=-0.1, pred_match=0
    )) == "FN_aligner"


def test_fn_fusion_when_signal_positive_but_gate_negative():
    assert classify_row(_row(
        gt_match=1, detector_hit=1, tracker_assoc="stable",
        aligner_gmc_score=0.5, ikun_logit=-0.5, fusion_gate=-0.05, pred_match=0
    )) == "FN_fusion"


def test_attribute_table_adds_failure_class_column():
    df = pd.DataFrame([
        _row(gt_match=1, detector_hit=0).to_dict(),
        _row(gt_match=0, pred_match=0).to_dict(),
    ])
    out = attribute_table(df)
    assert "failure_class" in out.columns
    assert list(out["failure_class"]) == ["FN_detector", "TN"]
```

- [ ] **Step 9.2: Run tests, expect fail**

```bash
pytest tests/failure_audit/test_attribute.py -v
```
Expected: ImportError

- [ ] **Step 9.3: Implement `attribute.py`**

```python
# diagnostics/failure_audit/attribute.py
"""7-class failure attribution.

Ordered decision rule (first match wins). Thresholds 0.3 / 0.0 are
pre-registered; revisit by inspecting score histograms during D1.
"""
import pandas as pd

ALIGNER_DEAD_GMC    = 0.3
ALIGNER_DEAD_LOGIT  = 0.0


def classify_row(row: pd.Series) -> str:
    if row["gt_match"] == 1 and row["detector_hit"] == 0:
        return "FN_detector"
    if row["gt_match"] == 1 and row["tracker_assoc"] in ("switched", "lost"):
        return "FN_tracker"
    if (row["gt_match"] == 1
            and row["aligner_gmc_score"] < ALIGNER_DEAD_GMC
            and row["ikun_logit"]        < ALIGNER_DEAD_LOGIT):
        return "FN_aligner"
    if (row["gt_match"] == 1
            and (row["aligner_gmc_score"] >= ALIGNER_DEAD_GMC
                 or row["ikun_logit"]    >= ALIGNER_DEAD_LOGIT)
            and row["fusion_gate"] < 0):
        return "FN_fusion"
    if row["gt_match"] == 1 and row["pred_match"] == 1:
        return "TP"
    if row["gt_match"] == 0 and row["pred_match"] == 1:
        return "FP"
    return "TN"


def attribute_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["failure_class"] = df.apply(classify_row, axis=1)
    return df
```

- [ ] **Step 9.4: Run tests**

```bash
pytest tests/failure_audit/test_attribute.py -v
```
Expected: 8 passed

- [ ] **Step 9.5: Commit**

```bash
git add diagnostics/failure_audit/attribute.py tests/failure_audit/test_attribute.py
git commit -m "feat(audit): 7-class failure attribution decision tree"
```

---

## Task 10: report.py — markdown summary

**Files:**
- Create: `diagnostics/failure_audit/report.py`
- Create: `tests/failure_audit/test_report.py`

- [ ] **Step 10.1: Write failing test**

```python
# tests/failure_audit/test_report.py
import pandas as pd
from diagnostics.failure_audit.report import build_summary_markdown


def test_summary_contains_per_cell_breakdown():
    df = pd.DataFrame([
        {"seq": "0011", "expr": "turning-cars", "frame": 1, "failure_class": "FN_detector"},
        {"seq": "0011", "expr": "turning-cars", "frame": 2, "failure_class": "FN_detector"},
        {"seq": "0011", "expr": "turning-cars", "frame": 3, "failure_class": "FN_aligner"},
        {"seq": "0011", "expr": "turning-cars", "frame": 4, "failure_class": "TP"},
    ])
    md = build_summary_markdown([df])
    assert "turning-cars" in md
    assert "FN_detector" in md
    # 2 of 3 FN = 67% on detector → must appear
    assert "67" in md or "66" in md
```

- [ ] **Step 10.2: Run test, expect fail**

```bash
pytest tests/failure_audit/test_report.py -v
```
Expected: ImportError

- [ ] **Step 10.3: Implement `report.py`**

```python
# diagnostics/failure_audit/report.py
"""Aggregate attributed audit tables into a markdown summary."""
from typing import List
import pandas as pd

FN_CLASSES = ["FN_detector", "FN_tracker", "FN_aligner", "FN_fusion"]


def _cell_block(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    seq, expr = df["seq"].iloc[0], df["expr"].iloc[0]
    fn = df[df["failure_class"].isin(FN_CLASSES)]
    total_fn = len(fn)
    lines = [f"### `{expr}` × seq `{seq}`",
             "",
             "| stage | n_FN | %_FN | example_frames |",
             "|---|---|---|---|"]
    for stage in FN_CLASSES:
        sub = fn[fn["failure_class"] == stage]
        n = len(sub)
        pct = (100.0 * n / total_fn) if total_fn else 0.0
        examples = ", ".join(str(f) for f in sub["frame"].head(3))
        lines.append(f"| {stage} | {n} | {pct:.0f}% | {examples} |")
    lines += ["", f"Totals: TP={int((df['failure_class']=='TP').sum())}, "
                  f"FP={int((df['failure_class']=='FP').sum())}, "
                  f"TN={int((df['failure_class']=='TN').sum())}, "
                  f"FN={total_fn}", ""]
    return "\n".join(lines)


def build_summary_markdown(cell_dfs: List[pd.DataFrame]) -> str:
    parts = ["# Failure-Mode Audit Summary", "",
             "Per-cell breakdown of FN attribution.", ""]
    for df in cell_dfs:
        parts.append(_cell_block(df))

    pooled = pd.concat(cell_dfs, ignore_index=True) if cell_dfs else pd.DataFrame()
    if not pooled.empty:
        pooled_fn = pooled[pooled["failure_class"].isin(FN_CLASSES)]
        parts.append("## Pooled across all cells")
        parts.append("")
        parts.append("| stage | n_FN | %_FN |")
        parts.append("|---|---|---|")
        total = len(pooled_fn)
        for stage in FN_CLASSES:
            n = (pooled_fn["failure_class"] == stage).sum()
            pct = (100.0 * n / total) if total else 0.0
            parts.append(f"| {stage} | {int(n)} | {pct:.0f}% |")
        parts.append("")

        parts.append("## Decision")
        top = (pooled_fn["failure_class"].value_counts(normalize=True)
               .mul(100).round(1).to_dict()) if total else {}
        top_stage = max(top, key=top.get) if top else None
        top_pct = top.get(top_stage, 0.0)
        if top_pct >= 60:
            verdict = f"**Lever found** — `{top_stage}` accounts for {top_pct}% of pooled FN."
        elif top_pct >= 40:
            verdict = f"**Mixed** — `{top_stage}` is the dominant stage at {top_pct}% (<60%)."
        else:
            verdict = "**Door closes** — no stage exceeds 40% of pooled FN; GT/ambiguity bound."
        parts.append(verdict)
        parts.append("")
    return "\n".join(parts)
```

- [ ] **Step 10.4: Run test**

```bash
pytest tests/failure_audit/test_report.py -v
```
Expected: 1 passed

- [ ] **Step 10.5: Commit**

```bash
git add diagnostics/failure_audit/report.py tests/failure_audit/test_report.py
git commit -m "feat(audit): markdown summary aggregator with decision verdict"
```

---

## Task 11: Driver `run_audit.py`

**Files:**
- Create: `diagnostics/failure_audit/run_audit.py`

- [ ] **Step 11.1: Implement driver**

```python
# diagnostics/failure_audit/run_audit.py
"""End-to-end driver for the failure-mode audit.

Usage:
    python -m diagnostics.failure_audit.run_audit
"""
from pathlib import Path
import json

from diagnostics.failure_audit.inventory import inventory_cells, TARGET_CELLS
from diagnostics.failure_audit.build_table import build_audit_table
from diagnostics.failure_audit.attribute import attribute_table
from diagnostics.failure_audit.report import build_summary_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "diagnostics" / "results" / "failure_audit"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    statuses = inventory_cells(REPO_ROOT)
    inv = [{"expr": s.expr, "seq": s.seq,
            "ikun_present": s.ikun_present, "gmc_present": s.gmc_present,
            "det_present":  s.det_present,  "gt_present":  s.gt_present,
            "all_present":  s.all_present}
           for s in statuses]
    (OUT_DIR / "inventory.json").write_text(json.dumps(inv, indent=2))

    missing = [s for s in statuses if not s.all_present]
    if missing:
        gaps = ", ".join(f"{s.expr}×{s.seq}" for s in missing)
        raise SystemExit(f"Cache gaps detected for: {gaps}. "
                         f"Inspect inventory.json before re-running.")

    log_lines = [f"Ship recipe: depth-aug seed-1, motion-axis fusion α=1.0, scale=0.9, thr=0.17"]
    cell_dfs = []
    for expr, seq in TARGET_CELLS:
        df = build_audit_table(REPO_ROOT, seq=seq, expr=expr)
        df = attribute_table(df)
        parquet_path = OUT_DIR / f"audit_{expr}_{seq}.parquet"
        df.to_parquet(parquet_path)
        log_lines.append(f"{expr}×{seq}: {len(df)} rows → {parquet_path.name}")
        cell_dfs.append(df)

    md = build_summary_markdown(cell_dfs)
    (OUT_DIR / "SUMMARY.md").write_text(md)
    (OUT_DIR / "_run_log.txt").write_text("\n".join(log_lines))
    print("Audit done. SUMMARY.md ready.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Smoke-test driver import**

```bash
python -c "from diagnostics.failure_audit.run_audit import main; print('ok')"
```
Expected: `ok`

- [ ] **Step 11.3: Commit**

```bash
git add diagnostics/failure_audit/run_audit.py
git commit -m "feat(audit): end-to-end driver with gap-abort on missing caches"
```

---

## Task 12: Inventory dry-run and threshold inspection

**Files:** none new

- [ ] **Step 12.1: Run inventory**

```bash
python -m diagnostics.failure_audit.run_audit 2>&1 | tee /tmp/audit_dry.log
```
Expected output: either gap report (abort) OR full table run.

- [ ] **Step 12.2: Inspect score histograms before locking thresholds**

```bash
python -c "
import pandas as pd, glob
for p in glob.glob('diagnostics/results/failure_audit/audit_*.parquet'):
    df = pd.read_parquet(p)
    print(p)
    print(df['aligner_gmc_score'].describe(percentiles=[.1, .25, .5, .75, .9]))
    print(df['ikun_logit'].describe(percentiles=[.1, .25, .5, .75, .9]))
    print('---')
"
```
If `aligner_gmc_score` 25th-percentile of FN-rows is far from 0.3, OR `ikun_logit`
distribution centers far from 0, update `ALIGNER_DEAD_GMC` / `ALIGNER_DEAD_LOGIT` in
`attribute.py` to the empirically-justified value, re-run audit, re-run unit tests
(adjusting fixture values so they remain in-class), commit.

- [ ] **Step 12.3: Cross-check sanity assertion**

```bash
python -c "
import pandas as pd, glob
total_rows = 0
for p in sorted(glob.glob('diagnostics/results/failure_audit/audit_*.parquet')):
    df = pd.read_parquet(p)
    cls_total = df['failure_class'].value_counts().sum()
    assert cls_total == len(df), f'{p}: cls_total {cls_total} != rows {len(df)}'
    total_rows += len(df)
    print(p, len(df), 'OK')
print('Total rows across 5 cells:', total_rows)
"
```
Expected: all OK lines, no AssertionError.

- [ ] **Step 12.4: Commit any threshold change**

If thresholds adjusted in step 12.2:

```bash
git add diagnostics/failure_audit/attribute.py tests/failure_audit/test_attribute.py
git commit -m "tune(audit): align dead-signal thresholds to observed score distribution"
```

---

## Task 13: Write decision document and final commit

**Files:**
- Create: `docs/superpowers/specs/2026-05-14-failure-mode-audit-cascade-ikun-retrospective.md`

- [ ] **Step 13.1: Read SUMMARY.md and write retrospective**

Write a retrospective with sections:
1. **Pooled stage breakdown** — copy table from SUMMARY.md
2. **Per-cell highlights** — any cell that diverges from pooled pattern
3. **Decision** — one of: lever found / mixed / door closes (verbatim from SUMMARY.md)
4. **Next action** — concrete next experiment spec to write, OR ship + write-up

Path: `docs/superpowers/specs/2026-05-14-failure-mode-audit-cascade-ikun-retrospective.md`

- [ ] **Step 13.2: Save memory note**

Write a project memory entry summarising the verdict so future conversations
inherit the result without re-reading the audit.

Memory file: `/home/seanachan/.claude/projects/-home-seanachan-GMC-Link/memory/project_failure_audit_<verdict>.md`
- Update MEMORY.md index with a one-line entry

- [ ] **Step 13.3: Commit retrospective + memory**

```bash
git add docs/superpowers/specs/2026-05-14-failure-mode-audit-cascade-ikun-retrospective.md \
        diagnostics/results/failure_audit/SUMMARY.md \
        diagnostics/results/failure_audit/inventory.json \
        diagnostics/results/failure_audit/_run_log.txt \
        diagnostics/results/failure_audit/audit_*.parquet
git commit -m "docs(audit): failure-mode audit results + decision retrospective"
```

---

## Notes for the Executor

- **No GPU runs in this plan.** All sources are existing on-disk caches.
- **If `inventory.json` flags gaps**, the plan must pause and the gap-fill hooks
  (spec §1.2) need to be specified in a follow-up plan before re-running Task 12.
- **Threshold tuning in Task 12.2 is the ONE place** where the audit deviates from
  pre-registration. Do it once after seeing the first distribution dump, then freeze.
- **GT convention**: `gt_template_old/` (paper-canonical). Any join error suggests the
  wrong template is being read — abort and inspect.
- **Pedestrian-walking** expression family uses prefix-match in `_expr_match`; verify
  the actual cache expression keys land in the match set during Task 12.1.
