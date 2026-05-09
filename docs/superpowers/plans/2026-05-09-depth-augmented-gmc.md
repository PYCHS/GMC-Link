# Depth-Augmented GMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4D depth features `[Z_n, dZ_2, dZ_5, dZ_10]` to existing 13D motion vector to break the 0.7793 micro AUC ceiling that 16 prior aligner-side levers couldn't.

**Architecture:** Depth Anything V2 metric-vkitti large via HuggingFace `transformers`; per-track Z time-series cache (~330MB total, NOT raw maps); 17D motion projector with zero-init weight cols `[13:17]` for bit-exact step-0 forward; ego-Z compensation via stationary-track median dZ; conda env `RMOT`; branch `exp/ego-motion-systematic`.

**Tech Stack:** Python 3.10, PyTorch 2.x, transformers (HuggingFace), Depth Anything V2 metric-vkitti-large, MiniLM lang encoder, sentence-transformers, NumPy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-05-09-depth-augmented-gmc-design.md`

---

## File Structure

### Create
- `gmc_link/depth_extractor.py` — Depth Anything V2 wrapper (`extract_depth_map(frame_bgr) → np.ndarray[H,W] float32 meters`).
- `gmc_link/depth_cache.py` — per-track Z cache load/lookup utilities (mirror `gmc_link/clip_cache.py` API).
- `run_build_depth_cache.py` — driver: iterate tracker `predict.txt` × seq, compute Z time-series, write `gmc_link/depth_cache/z_track_{arch}_{seq}.json`.
- `run_build_gmc_cache_depth.py` — eval-time gmc score cache builder using 17D-aligner.
- `tests/test_depth_extractor.py`, `tests/test_depth_cache.py`, `tests/test_alignment_motion_dim.py` — unit tests (TDD).

### Modify
- `gmc_link/manager.py` — `GMCLinkManager.process_frame` accepts optional `depth_z_lookup`. Append 4D depth features when present.
- `gmc_link/dataset.py` — load depth cache + emit 17D motion vector under `--use-depth`.
- `gmc_link/alignment.py` — `motion_dim` already kwargable; add identity-init logic for `[13:17]` cols when `motion_dim=17`.
- `gmc_link/train.py` — `--use-depth` + `--depth-cache-path` flags; checkpoint metadata `motion_dim`.
- `diagnostics/diag_gt_cosine_distributions.py` — read `motion_dim` from checkpoint metadata; pass to model construction; load depth cache if `use_depth` set.

### NO change to
- `gmc_link/losses.py`, `gmc_link/text_utils.py`, per-arch eval scripts (`run_ikun_linear_additive.py`, `run_flexhook_phase5_gmc_sweep.py`, `run_flexhook_v2_raw_sweep.py`).

---

## Task 1: `depth_extractor.py` Depth Anything V2 wrapper + smoke test

**Files:**
- Create: `gmc_link/depth_extractor.py`
- Create: `tests/test_depth_extractor.py`

**Why first:** Cheapest-to-fail step. If HF transformers route doesn't work or Depth Anything V2 outputs aren't metric meters, kill direction now (zero sunk cost).

- [ ] **Step 1.1: Write failing test**

```python
# tests/test_depth_extractor.py
import numpy as np
import pytest
from PIL import Image
from gmc_link.depth_extractor import DepthExtractor

KITTI_IMG = "/home/seanachan/data/Dataset/refer-kitti-v2/KITTI/training/image_02/0011/000100.png"

@pytest.fixture(scope="module")
def extractor():
    return DepthExtractor(device="cuda")

def test_output_shape_and_dtype(extractor):
    img = np.array(Image.open(KITTI_IMG).convert("RGB"))
    depth = extractor.extract(img)
    assert depth.dtype == np.float32
    assert depth.ndim == 2
    assert depth.shape == img.shape[:2]

def test_metric_range_kitti(extractor):
    img = np.array(Image.open(KITTI_IMG).convert("RGB"))
    depth = extractor.extract(img)
    valid = depth[(depth > 0) & (depth < 200)]
    assert valid.size > 0.5 * depth.size, "too many invalid depths"
    assert 3.0 < np.median(valid) < 80.0, f"median {np.median(valid)} not KITTI-plausible"

def test_patch_median(extractor):
    img = np.array(Image.open(KITTI_IMG).convert("RGB"))
    depth = extractor.extract(img)
    cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
    patch = depth[cy-2:cy+3, cx-2:cx+3]
    z = np.median(patch)
    assert 0.5 < z < 200.0
```

- [ ] **Step 1.2: Run test — verify fails (module not found)**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_depth_extractor.py -v
```

Expected: `ModuleNotFoundError: No module named 'gmc_link.depth_extractor'`

- [ ] **Step 1.3: Write `depth_extractor.py`**

```python
# gmc_link/depth_extractor.py
"""Depth Anything V2 metric-vkitti-large wrapper. One-shot per-frame inference.

Output: float32 metric depth map, meters, shape (H, W) matching input image.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

_HF_MODEL = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large-hf"


class DepthExtractor:
    def __init__(self, device: str = "cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(_HF_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            _HF_MODEL, torch_dtype=dtype
        ).to(device).eval()

    @torch.inference_mode()
    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        H, W = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device, dtype=self.dtype)
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth  # (1, h, w)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1).squeeze(0)
        return pred.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_batch(self, images_rgb: list[np.ndarray]) -> list[np.ndarray]:
        return [self.extract(im) for im in images_rgb]
```

- [ ] **Step 1.4: Run test — verify passes**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_depth_extractor.py -v
```

Expected: all 3 tests pass. If any fail, inspect the output range — Depth Anything V2 may emit relative units instead of meters; switch to `Metric3D-ViT-L` fallback per spec Risk row 1.

- [ ] **Step 1.5: Commit**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/depth_extractor.py tests/test_depth_extractor.py
git commit -m "$(cat <<'EOF'
feat(depth): Depth Anything V2 metric-vkitti-large wrapper

HuggingFace transformers route. Output float32 meters at input resolution.
Smoke test on KITTI seq 0011 frame 000100 verifies metric range.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `depth_cache.py` + `run_build_depth_cache.py` driver

**Files:**
- Create: `gmc_link/depth_cache.py`
- Create: `run_build_depth_cache.py`
- Create: `tests/test_depth_cache.py`

- [ ] **Step 2.1: Write failing test**

```python
# tests/test_depth_cache.py
import json
import tempfile
from pathlib import Path
from gmc_link.depth_cache import DepthCache, save_depth_cache

def test_round_trip(tmp_path: Path):
    data = {"42": {"100": 12.5, "101": 13.1}, "7": {"100": 5.5}}
    p = tmp_path / "z_track_ikun_0011.json"
    save_depth_cache(data, p)
    cache = DepthCache.load(p)
    assert cache.lookup(track_id=42, frame_id=100) == 12.5
    assert cache.lookup(track_id=7,  frame_id=100) == 5.5
    assert cache.lookup(track_id=99, frame_id=100) is None
    assert cache.lookup(track_id=42, frame_id=999) is None

def test_seq_lookup_normalises_str_keys(tmp_path: Path):
    p = tmp_path / "c.json"
    save_depth_cache({"1": {"5": 7.0}}, p)
    cache = DepthCache.load(p)
    assert cache.lookup(1, 5) == 7.0
    assert cache.lookup("1", "5") == 7.0
```

- [ ] **Step 2.2: Run test — verify fails (module not found)**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_depth_cache.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 2.3: Write `depth_cache.py`**

```python
# gmc_link/depth_cache.py
"""Per-track Z time-series cache. Format: {track_id_str: {frame_id_str: z_m}}."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def save_depth_cache(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(t): {str(f): float(z) for f, z in fmap.items()} for t, fmap in data.items()}
    path.write_text(json.dumps(serialisable))


@dataclass
class DepthCache:
    table: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: Path | str) -> "DepthCache":
        return cls(json.loads(Path(path).read_text()))

    def lookup(self, track_id, frame_id) -> Optional[float]:
        per_track = self.table.get(str(track_id))
        if per_track is None:
            return None
        return per_track.get(str(frame_id))
```

- [ ] **Step 2.4: Run test — verify passes**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_depth_cache.py -v
```

Expected: 2/2 pass.

- [ ] **Step 2.5: Write `run_build_depth_cache.py` driver**

```python
# run_build_depth_cache.py
"""Build per-track Z cache for one (arch, seq).

Usage:
  python run_build_depth_cache.py --arch ikun --seq 0011
  python run_build_depth_cache.py --arch ikun --seq 0005 0011 0013
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from gmc_link.depth_extractor import DepthExtractor
from gmc_link.depth_cache import save_depth_cache

KITTI_IMG_DIR = Path("/home/seanachan/data/Dataset/refer-kitti-v2/KITTI/training/image_02")
NS_DIR = Path("/home/seanachan/GMC-Link/NeuralSORT")


def load_tracks(arch: str, seq: str) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """{frame_id: [(track_id, x, y, w, h), ...]}"""
    tracks: dict[int, list] = {}
    for cls in ("car", "pedestrian"):
        f = NS_DIR / seq / cls / "predict.txt"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            tracks.setdefault(frame_id, []).append((track_id, x, y, w, h))
    return tracks


def patch_median(depth: np.ndarray, cx: int, cy: int, half: int = 2) -> float:
    H, W = depth.shape
    cx = int(np.clip(cx, half, W - 1 - half))
    cy = int(np.clip(cy, half, H - 1 - half))
    patch = depth[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return float(np.median(patch))


def build_one(arch: str, seq: str, extractor: DepthExtractor, out_dir: Path) -> None:
    tracks = load_tracks(arch, seq)
    img_dir = KITTI_IMG_DIR / seq
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    z_table: dict[int, dict[int, float]] = {}
    frame_ids = sorted(tracks.keys())
    for frame_id in tqdm(frame_ids, desc=f"{arch}/{seq}"):
        img_path = img_dir / f"{frame_id:06d}.png"
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        depth = extractor.extract(img)
        depth = np.clip(depth, 0, 80)
        for tid, x, y, w, h in tracks[frame_id]:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            z_table.setdefault(tid, {})[frame_id] = patch_median(depth, cx, cy)
    out_path = out_dir / f"z_track_{arch}_{seq}.json"
    save_depth_cache(z_table, out_path)
    print(f"wrote {out_path} ({len(z_table)} tracks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ikun", "fh_v1", "fh_v2"])
    ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--out-dir", default="gmc_link/depth_cache")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    extractor = DepthExtractor(device="cuda")
    for s in args.seq:
        build_one(args.arch, s, extractor, out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.6: Smoke-test driver on 1 seq**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python run_build_depth_cache.py --arch ikun --seq 0011
```

Expected: tqdm progress bar, output `wrote gmc_link/depth_cache/z_track_ikun_0011.json (~50 tracks)`. Wall time ~5min on H100.

Sanity check the cache:

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -c "
from gmc_link.depth_cache import DepthCache
c = DepthCache.load('gmc_link/depth_cache/z_track_ikun_0011.json')
print(f'tracks={len(c.table)}')
for tid in list(c.table.keys())[:3]:
    fids = sorted(int(f) for f in c.table[tid].keys())
    print(f'  track {tid}: frames {fids[0]}..{fids[-1]} ({len(fids)} samples), Z range {min(c.table[tid].values()):.1f}..{max(c.table[tid].values()):.1f}m')
"
```

Expected: 30-100 tracks, Z values 3-80m, no NaN/Inf.

- [ ] **Step 2.7: Commit**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/depth_cache.py run_build_depth_cache.py tests/test_depth_cache.py gmc_link/depth_cache/z_track_ikun_0011.json
git commit -m "$(cat <<'EOF'
feat(depth): per-track Z cache + build driver

Cache format: {track_id: {frame_id: z_m}}, JSON, ~50 tracks/seq, ~5MB total/arch.
Driver iterates NeuralSORT predict.txt × KITTI image_02, samples 5x5 patch median
at bbox center, clips Z to [0,80]. Smoke-built ikun/0011.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Build full depth cache (3 archs × all eval seqs)

**Files:** none modified — driver invocation only. Outputs to `gmc_link/depth_cache/`.

**Why:** One-shot pre-compute now so all subsequent training/eval loops are cache hits.

- [ ] **Step 3.1: Build iKUN test seqs**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python run_build_depth_cache.py --arch ikun --seq 0005 0013
```

Expected: `wrote z_track_ikun_0005.json`, `wrote z_track_ikun_0013.json`. Wall time ~10min total.

- [ ] **Step 3.2: Build FH V1 test seqs**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python run_build_depth_cache.py --arch fh_v1 --seq 0005 0011 0013
```

Note: FH V1 reuses NeuralSORT trackers same as iKUN, so the cache files differ only in `arch` filename suffix when track-IDs match. To save time, you may symlink:

```bash
cd gmc_link/depth_cache && for s in 0005 0011 0013; do ln -sf z_track_ikun_$s.json z_track_fh_v1_$s.json; done
```

- [ ] **Step 3.3: Build FH V2 test seqs (incl 0019)**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python run_build_depth_cache.py --arch fh_v2 --seq 0005 0011 0013 0019
```

Wall time ~25 min (V2 0019 is the long seq).

- [ ] **Step 3.4: Build V1 train seqs (15 seqs)**

Needed for stage1 training samples. Train seqs IDs from `gmc_link/train.py`: `0001,0002,0003,0004,0006,0007,0008,0009,0010,0012,0014,0015,0016,0018,0020`.

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python run_build_depth_cache.py --arch ikun --seq 0001 0002 0003 0004 0006 0007 0008 0009 0010 0012 0014 0015 0016 0018 0020
```

Wall time ~2-3hr H100. Run in tmux/background.

- [ ] **Step 3.5: Verify cache coverage**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -c "
import json
from pathlib import Path
total_tracks = 0
total_samples = 0
for p in sorted(Path('gmc_link/depth_cache').glob('z_track_*.json')):
    if p.is_symlink(): continue
    d = json.loads(p.read_text())
    n_tr = len(d)
    n_s = sum(len(v) for v in d.values())
    print(f'{p.name}: {n_tr} tracks, {n_s} samples')
    total_tracks += n_tr; total_samples += n_s
print(f'TOTAL: {total_tracks} tracks, {total_samples} samples')
"
```

Expected: total samples ~50k-150k. KILL if any cache file has 0 tracks (driver bug).

- [ ] **Step 3.6: Commit cache files**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/depth_cache/*.json
git commit -m "$(cat <<'EOF'
feat(depth): pre-built Z caches for V1 train + V1/V2 test

3 archs × {15 train + 3-4 test} seqs. ~330MB total. Drives 17D motion vector.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `manager.py` 17D motion vector path

**Files:**
- Modify: `gmc_link/manager.py` (`GMCLinkManager.process_frame` and `__init__`)
- Modify: `tests/test_manager_17d.py` (new test file)

- [ ] **Step 4.1: Read current 13D builder location**

```bash
cd /home/seanachan/GMC-Link && grep -n "spatial_motion = np" gmc_link/manager.py
```

Note the line number for the 13D `np.array([dx_s, dy_s, dx_m, dy_m, dx_l, dy_l, dw, dh, cx, cy, w, h, snr])` construction.

- [ ] **Step 4.2: Write failing test**

```python
# tests/test_manager_17d.py
import numpy as np
from gmc_link.manager import GMCLinkManager

def test_13d_default():
    m = GMCLinkManager()
    assert m.motion_dim == 13

def test_17d_with_depth_lookup():
    m = GMCLinkManager(use_depth=True)
    assert m.motion_dim == 17

def test_dz_residual_with_ego_compensation():
    """dZ_residual = dZ_track − median(dZ over stationary tracks)."""
    m = GMCLinkManager(use_depth=True)
    z_now = {1: 20.0, 2: 30.0, 3: 40.0}
    z_prev = {1: 19.0, 2: 29.0, 3: 39.0}  # all 3 closer by 1m (ego forward)
    stationary_ids = {1, 2, 3}
    dz_compensated = m._compute_dz_residual(z_now, z_prev, stationary_ids)
    for tid in (1, 2, 3):
        assert abs(dz_compensated[tid] - 0.0) < 1e-5  # ego-comp removes uniform shift

def test_dz_residual_isolates_approaching_object():
    m = GMCLinkManager(use_depth=True)
    z_now = {1: 20.0, 2: 30.0, 3: 35.0}  # tid=3 was 39, now 35 (approaching by 4m)
    z_prev = {1: 19.0, 2: 29.0, 3: 39.0}
    stationary_ids = {1, 2}
    dz = m._compute_dz_residual(z_now, z_prev, stationary_ids)
    assert abs(dz[1] - 0.0) < 1e-5
    assert abs(dz[2] - 0.0) < 1e-5
    assert abs(dz[3] - (-5.0)) < 1e-5  # 35-39=-4 minus ego_dz=+1 → -5
```

- [ ] **Step 4.3: Run — verify fails**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_manager_17d.py -v
```

Expected: TypeError or AttributeError on `use_depth` / `_compute_dz_residual`.

- [ ] **Step 4.4: Implement `motion_dim` + dZ helper in `manager.py`**

In `GMCLinkManager.__init__`, after the existing init, add:

```python
# 17D depth path
self.use_depth = bool(kwargs.get("use_depth", False))
self.motion_dim = 17 if self.use_depth else 13
# per-track Z history at FRAME_GAPS
self._z_history: dict[int, list[tuple[int, float]]] = {}  # {track_id: [(frame_id, z), ...]}
```

Add helper method on the class:

```python
def _compute_dz_residual(
    self,
    z_now: dict[int, float],
    z_prev: dict[int, float],
    stationary_ids: set[int],
) -> dict[int, float]:
    """Per-track dZ with ego-Z compensation.

    dZ_ego = median(dZ) over stationary tracks; dZ_residual = dZ_track − dZ_ego.
    Falls back to zero compensation if zero stationary tracks.
    """
    dz_raw = {tid: z_now[tid] - z_prev[tid] for tid in z_now if tid in z_prev}
    stat_dz = [dz_raw[tid] for tid in stationary_ids if tid in dz_raw]
    if stat_dz:
        dz_ego = float(np.median(stat_dz))
    else:
        dz_ego = 0.0
    return {tid: v - dz_ego for tid, v in dz_raw.items()}
```

In `process_frame`, after building 13D `spatial_motion` and after computing residual velocities, build the 4D depth slice:

```python
if self.use_depth and depth_z_lookup is not None:
    track_id = ...  # already known per-track in the loop
    z_now = depth_z_lookup.get(track_id)
    if z_now is None:
        depth_4d = np.zeros(4, dtype=np.float32)
    else:
        z_n = np.clip(z_now, 0, 80) / 100.0
        # pull history at gaps 2/5/10
        hist = self._z_history.get(track_id, [])
        z_at_gap = {g: None for g in self.FRAME_GAPS}
        for past_fid, past_z in reversed(hist):
            gap = self.frame_id - past_fid
            for g in self.FRAME_GAPS:
                if z_at_gap[g] is None and gap >= g:
                    z_at_gap[g] = past_z
        # ego-Z residual: needs cohort. Compute once per frame outside per-track loop.
        # Here we use raw dZ per scale; ego comp applied at frame level (see batched path).
        dz = []
        for g in self.FRAME_GAPS:
            if z_at_gap[g] is None:
                dz.append(0.0)
            else:
                dz.append((z_now - z_at_gap[g]) / 10.0)
        depth_4d = np.array([z_n, dz[0], dz[1], dz[2]], dtype=np.float32)
        # update history with current
        self._z_history.setdefault(track_id, []).append((self.frame_id, z_now))
        # cap history to longest gap
        max_gap = max(self.FRAME_GAPS)
        self._z_history[track_id] = [
            (f, z) for f, z in self._z_history[track_id] if self.frame_id - f <= max_gap + 1
        ]
    spatial_motion = np.concatenate([spatial_motion, depth_4d]).astype(np.float32)
```

Update `process_frame` signature to accept `depth_z_lookup: dict[int, float] | None = None`.

For ego-Z compensation at frame level (across all tracks): apply once per frame on the cohort dZ_2/dZ_5/dZ_10 values BEFORE per-track packing. Simplest path: recompute `dz_ego_per_gap` at the start of the per-track loop based on `stationary_ids = {tid: ||res_v|| < 1 px/frame}`, then subtract from each track's dz. Add this in the `process_frame` body before the per-track loop.

- [ ] **Step 4.5: Run unit tests — verify passes**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_manager_17d.py -v
```

Expected: 4/4 pass.

- [ ] **Step 4.6: Backward-compat smoke test**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -c "
from gmc_link.manager import GMCLinkManager
m = GMCLinkManager()  # default 13D path
print(f'motion_dim={m.motion_dim}, use_depth={m.use_depth}')
assert m.motion_dim == 13
assert m.use_depth is False
"
```

Expected: `motion_dim=13, use_depth=False`.

- [ ] **Step 4.7: Commit**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/manager.py tests/test_manager_17d.py
git commit -m "$(cat <<'EOF'
feat(manager): 17D motion vector path with ego-Z compensation

GMCLinkManager(use_depth=True) → motion_dim=17, appends [Z_n, dZ_2, dZ_5, dZ_10]
to existing 13D. Stationary-track median dZ used as ego-Z compensation per spec.
13D default path unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `dataset.py` emit 17D under `--use-depth`

**Files:**
- Modify: `gmc_link/dataset.py`
- Create: `tests/test_dataset_17d.py`

- [ ] **Step 5.1: Locate per-sample base_vec construction**

```bash
cd /home/seanachan/GMC-Link && grep -n "base_vec" gmc_link/dataset.py | head -20
```

Note line numbers around `_generate_positive_pairs`.

- [ ] **Step 5.2: Write failing test**

```python
# tests/test_dataset_17d.py
import json, tempfile
from pathlib import Path
import numpy as np
from gmc_link.dataset import build_training_data

def test_17d_motion_vector_when_use_depth(tmp_path: Path):
    """Smoke: build_training_data emits 17D when use_depth=True."""
    real_cache = Path("/home/seanachan/GMC-Link/gmc_link/depth_cache")
    motion_vecs, *_ = build_training_data(
        seqs=["0011"],
        use_depth=True,
        depth_cache_dir=real_cache,
    )
    assert motion_vecs.shape[1] == 17, f"expected 17D got {motion_vecs.shape}"

def test_13d_default_no_depth():
    motion_vecs, *_ = build_training_data(seqs=["0001"], use_depth=False)
    assert motion_vecs.shape[1] == 13
```

- [ ] **Step 5.3: Run — verify fails**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_dataset_17d.py -v
```

Expected: TypeError on unrecognized `use_depth` kwarg.

- [ ] **Step 5.4: Implement `--use-depth` plumb in `dataset.py`**

In `build_training_data` signature add:

```python
def build_training_data(
    ...,
    use_depth: bool = False,
    depth_cache_dir: Optional[Path] = None,
):
```

After loading raw sequence frames, before per-track motion vector emission, load the depth cache lazily:

```python
depth_caches: dict[str, DepthCache] = {}
if use_depth:
    from gmc_link.depth_cache import DepthCache
    arch = "ikun"  # train data uses ikun NS tracker by convention
    for seq in seqs:
        p = depth_cache_dir / f"z_track_{arch}_{seq}.json"
        if not p.exists():
            raise FileNotFoundError(f"depth cache missing: {p}")
        depth_caches[seq] = DepthCache.load(p)
```

**Frame-level ego-Z compensation at training time** (matches eval-time manager logic per spec):

For each `(seq, frame_id)` cohort, BEFORE per-track packing:

```python
# Per-frame cohort: compute ego dZ per gap from stationary tracks (||res_v|| < 1 px/frame)
# stationary criterion: residual velocity magnitude in normalized scale < 0.01 (== 1 px/frame after VELOCITY_SCALE=100)
def _frame_cohort_dz_ego(cache, frame_id, all_track_ids, residual_vels, gap):
    raw_dz = []
    for tid in all_track_ids:
        z_now = cache.lookup(tid, frame_id)
        z_prev = cache.lookup(tid, frame_id - gap)
        if z_now is None or z_prev is None:
            continue
        if np.linalg.norm(residual_vels.get(tid, [0, 0])) < 0.01:  # stationary
            raw_dz.append(z_now - z_prev)
    return float(np.median(raw_dz)) if raw_dz else 0.0
```

Then per-track:

```python
if use_depth:
    cache = depth_caches[seq]
    z_now = cache.lookup(track_id, frame_id)
    if z_now is None:
        depth_4d = np.zeros(4, dtype=np.float32)
    else:
        z_now = float(np.clip(z_now, 0, 80))
        z_n = z_now / 100.0
        dz_residual = []
        for g in (2, 5, 10):
            z_past = cache.lookup(track_id, frame_id - g)
            if z_past is None:
                dz_residual.append(0.0)
                continue
            raw_dz = z_now - z_past
            dz_ego = _frame_cohort_dz_ego(cache, frame_id, all_track_ids_in_frame, frame_residual_vels, g)
            dz_residual.append((raw_dz - dz_ego) / 10.0)
        depth_4d = np.array([z_n] + dz_residual, dtype=np.float32)
    base_vec = np.concatenate([base_vec, depth_4d]).astype(np.float32)
```

Cache `dz_ego` per `(seq, frame_id, gap)` to avoid recomputing for every track in the same frame.

Cache key for the prebuilt training cache must include `use_depth`:

```python
cache_key_parts.append(f"depth{int(use_depth)}")
```

- [ ] **Step 5.5: Run tests — verify passes**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_dataset_17d.py -v
```

Expected: 2/2 pass.

- [ ] **Step 5.6: Commit**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/dataset.py tests/test_dataset_17d.py
git commit -m "$(cat <<'EOF'
feat(dataset): 17D training samples under --use-depth

build_training_data(use_depth=True, depth_cache_dir=...) loads per-track Z cache,
appends [Z_n, dZ_2, dZ_5, dZ_10] to 13D motion vector. Cache key bumped to invalidate
prior 13D caches.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `alignment.py` motion_dim plumbing + identity-init + bit-exact verification

**Files:**
- Modify: `gmc_link/alignment.py` (already accepts `motion_dim` kwarg — add identity-init logic)
- Create: `tests/test_alignment_motion_dim.py`

- [ ] **Step 6.1: Confirm existing `motion_dim` plumb**

```bash
cd /home/seanachan/GMC-Link && grep -n "motion_dim" gmc_link/alignment.py | head -10
```

Expected: `motion_dim` already in `__init__` kwargs (default 13). Identity-init for new dims is the only new logic.

- [ ] **Step 6.2: Write failing test (bit-exact identity gate)**

```python
# tests/test_alignment_motion_dim.py
import torch
import numpy as np
from gmc_link.alignment import MotionLanguageAligner

def test_motion_dim_default_13():
    m = MotionLanguageAligner()
    assert m.motion_projector[0].in_features == 13

def test_motion_dim_17_changes_input_layer():
    m = MotionLanguageAligner(motion_dim=17)
    assert m.motion_projector[0].in_features == 17

def test_identity_init_zero_in_new_cols():
    """When motion_dim=17 with identity_init=True, weight cols [13:17] must be zero."""
    m = MotionLanguageAligner(motion_dim=17, identity_init_extra=True)
    W = m.motion_projector[0].weight  # (128, 17)
    assert torch.allclose(W[:, 13:17], torch.zeros_like(W[:, 13:17])), "extra cols not zero-init"
    # cols [:13] should be standard PyTorch init, NOT zero
    assert not torch.allclose(W[:, :13], torch.zeros_like(W[:, :13])), "13D cols accidentally zeroed"

def test_bit_exact_step0_with_zero_extra():
    """At step 0 with identity-init, 17D forward on [13D | zeros] == 13D forward on 13D."""
    torch.manual_seed(0)
    m13 = MotionLanguageAligner(motion_dim=13)
    torch.manual_seed(0)
    m17 = MotionLanguageAligner(motion_dim=17, identity_init_extra=True)
    # Match the first 13 cols by copying — since rng was reset, 13D init paths align,
    # but motion_projector[0] has shape (128, 13) vs (128, 17). Manually align:
    m17.motion_projector[0].weight.data[:, :13] = m13.motion_projector[0].weight.data
    m17.motion_projector[0].bias.data = m13.motion_projector[0].bias.data
    for src, dst in zip(m13.motion_projector[1:], m17.motion_projector[1:]):
        if hasattr(src, "weight") and src.weight is not None:
            dst.weight.data = src.weight.data.clone()
        if hasattr(src, "bias") and src.bias is not None:
            dst.bias.data = src.bias.data.clone()
    m17.lang_projector.load_state_dict(m13.lang_projector.state_dict())

    motion13 = torch.randn(8, 13)
    motion17 = torch.cat([motion13, torch.zeros(8, 4)], dim=-1)
    lang = torch.randn(8, 384)

    m13.eval(); m17.eval()
    with torch.inference_mode():
        score13 = m13(motion13, lang)
        score17 = m17(motion17, lang)

    assert torch.allclose(score13, score17, atol=1e-5), \
        f"max diff {(score13-score17).abs().max().item()}"
```

- [ ] **Step 6.3: Run — verify fails (identity_init_extra unrecognized)**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_alignment_motion_dim.py -v
```

Expected: TypeError on `identity_init_extra` kwarg or AssertionError on zero check.

- [ ] **Step 6.4: Implement identity-init**

In `MotionLanguageAligner.__init__`, after building `motion_projector` Sequential:

```python
def __init__(self, ..., motion_dim: int = 13, identity_init_extra: bool = False, ...):
    ...
    self.motion_dim = motion_dim
    # ... existing motion_projector build with Linear(motion_dim, 128) as [0] ...

    if identity_init_extra and motion_dim > 13:
        # zero-init the extra weight cols so step-0 forward ignores depth dims
        with torch.no_grad():
            self.motion_projector[0].weight[:, 13:].zero_()
        # bias unchanged — no contribution from extra dims
```

- [ ] **Step 6.5: Run tests — verify passes**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m pytest tests/test_alignment_motion_dim.py -v
```

Expected: 4/4 pass. The bit-exact test is the highest-value gate — if it fails, plumbing is wrong.

- [ ] **Step 6.6: Commit**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/alignment.py tests/test_alignment_motion_dim.py
git commit -m "$(cat <<'EOF'
feat(alignment): identity-init for motion_dim=17

MotionLanguageAligner(motion_dim=17, identity_init_extra=True) zeros the
weight cols [13:17] of motion_projector[0]. Step-0 forward bit-exact w.r.t.
13D aligner when extra dims are zeros. Plumbing-bug detector.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `train.py` `--use-depth` flag + stage1 train 3 seeds

**Files:**
- Modify: `gmc_link/train.py`
- Create: `run_train_depth.sh` (driver)

- [ ] **Step 7.1: Locate CLI block + checkpoint metadata save**

```bash
cd /home/seanachan/GMC-Link && grep -n "argparse\|add_argument\|motion_dim\|--use-clip-feat" gmc_link/train.py | head -25
```

Note `--use-clip-feat` line — mirror its plumbing pattern for `--use-depth`.

- [ ] **Step 7.2: Add CLI flags**

In the argparse block, add:

```python
ap.add_argument("--use-depth", action="store_true",
                help="Augment 13D motion vector with [Z_n, dZ_2, dZ_5, dZ_10] = 17D")
ap.add_argument("--depth-cache-path", type=str, default="gmc_link/depth_cache",
                help="Directory containing z_track_{arch}_{seq}.json")
```

In `_run_single_stage`, plumb to dataset + model:

```python
training_data = build_training_data(
    seqs=seqs,
    ...,
    use_depth=args.use_depth,
    depth_cache_dir=Path(args.depth_cache_path) if args.use_depth else None,
)

motion_dim = 17 if args.use_depth else 13
# verify cache produced expected shape
assert training_data[0].shape[1] == motion_dim, \
    f"dataset emitted {training_data[0].shape[1]}D but motion_dim={motion_dim}"

model = MotionLanguageAligner(
    motion_dim=motion_dim,
    identity_init_extra=args.use_depth,
    ...,
)
```

In checkpoint save block, store depth metadata:

```python
torch.save({
    "model": model.state_dict(),
    "motion_dim": motion_dim,
    "use_depth": args.use_depth,
    ...,
}, ckpt_path)
```

- [ ] **Step 7.3: Driver script `run_train_depth.sh`**

```bash
#!/bin/bash
# Stage1 17D depth-augmented aligner, 3 seeds.
set -e
cd /home/seanachan/GMC-Link
mkdir -p experiments/depth_v1train

for SEED in 0 1 2; do
  echo "=== seed $SEED ==="
  conda run -n RMOT python -m gmc_link.train \
    --use-depth \
    --depth-cache-path gmc_link/depth_cache \
    --seed $SEED \
    --epochs 100 --batch 256 \
    --output-dir experiments/depth_v1train/seed${SEED} \
    2>&1 | tee experiments/depth_v1train/seed${SEED}/train.log
done

echo "=== done. ckpts in experiments/depth_v1train/seed{0,1,2}/best.pth"
```

```bash
chmod +x run_train_depth.sh
```

- [ ] **Step 7.4: Smoke train 1 epoch to validate plumbing**

```bash
cd /home/seanachan/GMC-Link && conda run -n RMOT python -m gmc_link.train \
  --use-depth --depth-cache-path gmc_link/depth_cache \
  --seed 0 --epochs 1 --batch 256 \
  --output-dir /tmp/depth_smoke 2>&1 | tee /tmp/depth_smoke.log
```

Expected: training proceeds, no shape errors. Check `/tmp/depth_smoke/best.pth` saved with `motion_dim=17`:

```bash
conda run -n RMOT python -c "
import torch
ckpt = torch.load('/tmp/depth_smoke/best.pth', map_location='cpu')
print('keys:', list(ckpt.keys()))
print('motion_dim:', ckpt.get('motion_dim'))
print('use_depth:', ckpt.get('use_depth'))
W = ckpt['model']['motion_projector.0.weight']
print('motion_projector[0].weight shape:', tuple(W.shape))
"
```

Expected: `motion_dim: 17`, weight shape `(128, 17)`.

- [ ] **Step 7.5: Full 3-seed train**

```bash
cd /home/seanachan/GMC-Link && bash run_train_depth.sh
```

Wall time ~6hr H100. Run in tmux/background.

- [ ] **Step 7.6: Diag eval all 3 seeds**

Modify `diagnostics/diag_gt_cosine_distributions.py` to read `motion_dim`/`use_depth` from ckpt metadata + load depth cache when `use_depth=True`:

```python
ckpt = torch.load(args.weights, map_location="cpu")
motion_dim = ckpt.get("motion_dim", 13)
use_depth = ckpt.get("use_depth", False)
# ... existing model construction ...
model = MotionLanguageAligner(motion_dim=motion_dim, ...).to(device)
model.load_state_dict(ckpt["model"])
# at eval-data construction time, plumb use_depth through to dataset builder
```

Then run:

```bash
for SEED in 0 1 2; do
  conda run -n RMOT python diagnostics/diag_gt_cosine_distributions.py \
    --weights experiments/depth_v1train/seed${SEED}/best.pth \
    --seqs 0005 0011 0013 \
    --tag depth_seed${SEED} \
    2>&1 | tee experiments/depth_v1train/seed${SEED}/diag_eval.log
done
```

Expected: per-seed micro AUC pool. Compare to spec decision gate:

| micro AUC | Action |
|---|---|
| < 0.760 | KILL → Task 8 retrospective only |
| ∈ [0.760, 0.7793) | Marginal → run extended 5-seed |
| ∈ [0.7793, 0.79) | Proceeds to Step 4 (HOTA) |
| ≥ 0.79 | Strong POS → prioritize HOTA |

- [ ] **Step 7.7: Commit**

```bash
cd /home/seanachan/GMC-Link
git add gmc_link/train.py run_train_depth.sh diagnostics/diag_gt_cosine_distributions.py experiments/depth_v1train
git commit -m "$(cat <<'EOF'
feat(train): --use-depth stage1 path + 3-seed driver

train.py plumbs use_depth through dataset → 17D motion vector → aligner with
identity_init_extra=True. Checkpoint stores motion_dim + use_depth metadata for
backward-compat eval. diag eval reads metadata.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: HOTA cross-arch eval + ego-Z ablation + retrospective

**Files:**
- Create: `run_build_gmc_cache_depth.py` (eval-time gmc cache builder)
- Create: `run_depth_eval.sh` (driver)
- Create: `docs/superpowers/specs/2026-05-09-depth-augmented-gmc-retrospective.md`
- Memory: `project_depth_augmented_gmc_{positive,negative}.md` + `MEMORY.md` index entry

**Gate:** Only proceed past Step 8.1 if Task 7.6 micro AUC ≥ 0.760. If KILL, jump to Step 8.5 (retrospective only).

- [ ] **Step 8.1: GMC cache builder for 17D aligner**

Copy `run_build_gmc_cache.py` → `run_build_gmc_cache_depth.py`. Apply exactly these 3 edits (rest of file identical):

**Edit 1: imports** — add at top:

```python
from gmc_link.depth_cache import DepthCache
```

**Edit 2: model construction** — where the existing script does `MotionLanguageAligner()`:

```python
ckpt = torch.load(weights, map_location="cpu")
motion_dim = ckpt.get("motion_dim", 13)
assert motion_dim == 17, f"this script requires 17D ckpt, got {motion_dim}"
model = MotionLanguageAligner(motion_dim=17).cuda()
model.load_state_dict(ckpt["model"])
model.eval()
```

**Edit 3: GMCLinkManager + per-frame depth lookup** — where the per-seq loop runs `manager = GMCLinkManager()`:

```python
manager = GMCLinkManager(use_depth=True)
depth_cache = DepthCache.load(f"gmc_link/depth_cache/z_track_{args.arch}_{seq}.json")
# inside per-frame loop, before manager.process_frame:
frame_z_lookup = {tid: depth_cache.lookup(tid, frame_id) for tid in track_ids_in_frame}
frame_z_lookup = {k: v for k, v in frame_z_lookup.items() if v is not None}
# pass through:
result = manager.process_frame(..., depth_z_lookup=frame_z_lookup)
```

**Edit 4: output filename** — change cache filename suffix from `_cache.json` to `_depth_cache.json`:

```python
out_path = f"gmc_link/gmc_scores_{arch_tag}_{seq}_depth_cache.json"
```

The per-arch eval scripts (`run_ikun_linear_additive.py` etc.) read cache by `GMC_SUFFIX` env var — `export GMC_SUFFIX="_depth"` in Step 8.3 driver routes them to the new caches.

- [ ] **Step 8.2: Build depth gmc caches for 3 archs**

```bash
cd /home/seanachan/GMC-Link
W=experiments/depth_v1train/seed0/best.pth

GMC_WEIGHTS=$W conda run -n RMOT python run_build_gmc_cache_depth.py --arch ikun 0005 0011 0013
GMC_WEIGHTS=$W conda run -n RMOT python run_build_gmc_cache_depth.py --arch fh_v1 0005 0011 0013
GMC_WEIGHTS=$W conda run -n RMOT python run_build_gmc_cache_depth.py --arch fh_v2 0005 0011 0013 0019
```

Wall time ~3hr.

- [ ] **Step 8.3: HOTA pool eval at locked Arm A recipes**

```bash
# run_depth_eval.sh
#!/bin/bash
set -e
cd /home/seanachan/GMC-Link
RES=depth_eval_results.txt
echo "arch pool" > "$RES"
export GMC_SUFFIX="_depth"

# iKUN ship recipe
conda run -n RMOT python run_ikun_linear_additive.py \
  --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
  --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
  2>&1 | tee /tmp/ikun_depth.log
echo "ikun $(grep -oP 'pooled=\K[0-9.]+' /tmp/ikun_depth.log | tail -1)" >> "$RES"

# FH V1 ship recipe
conda run -n RMOT python run_flexhook_phase5_gmc_sweep.py \
  --alpha 0.65 --gmc_scale 10 --thr 3.0 \
  --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
  2>&1 | tee /tmp/fh_v1_depth.log
echo "fh_v1 $(grep -oP 'pooled=\K[0-9.]+' /tmp/fh_v1_depth.log | tail -1)" >> "$RES"

# FH V2 ship recipe
conda run -n RMOT python run_flexhook_v2_raw_sweep.py \
  --alpha 0.40 --gmc_scale 10 --thr 1.3 \
  --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
  2>&1 | tee /tmp/fh_v2_depth.log
echo "fh_v2 $(grep -oP 'pooled=\K[0-9.]+' /tmp/fh_v2_depth.log | tail -1)" >> "$RES"

cat "$RES"
echo
echo "vs Arm A multi-seed n=3:"
echo "  iKUN  : 44.608 ± 0.024"
echo "  FH V1 : 53.716 ± 0.068"
echo "  FH V2 : 42.799 ± 0.047"
echo "Ship gates (≥2σ):"
echo "  iKUN  ≥ 44.656"
echo "  FH V1 ≥ 53.852"
echo "  FH V2 ≥ 42.893"
```

```bash
chmod +x run_depth_eval.sh && bash run_depth_eval.sh
```

Decision per arch:

| Pool Δ vs Arm A | Status |
|---|---|
| ≥ +2σ | Multi-seed n=3 confirm + ship |
| ∈ [+0.05, +2σ) | Marginal — extend n=5 |
| ∈ [−0.05, +0.05] | NEUTRAL — document |
| < −0.05 | NEG — try recipe resweep `(α, sc, thr)` once before kill |

- [ ] **Step 8.4: Ego-Z ablation (Step 6 of spec)**

Re-run Stage1 with `dZ_ego = 0` (raw `dZ_track`, no compensation). Add a `--no-ego-z-comp` flag to `manager.py` `process_frame` cohort step + thread through `train.py`. Train 1 seed:

```bash
conda run -n RMOT python -m gmc_link.train \
  --use-depth --no-ego-z-comp \
  --seed 0 --epochs 100 --batch 256 \
  --output-dir experiments/depth_no_ego_comp/seed0
```

Diag eval; compare micro AUC vs the with-comp seed-0 result. Decision: if without-comp ≥ with-comp, ego-Z compensation is not the lever (raw Z + raw dZ alone is enough); if with-comp wins by ≥0.01 AUC, ego-Z compensation is a critical sub-lever.

- [ ] **Step 8.5: Retrospective doc**

Write `docs/superpowers/specs/2026-05-09-depth-augmented-gmc-retrospective.md` with:

- Status (POS/NEG/NEUTRAL per arch)
- Stage1 micro AUC table (3 seeds, mean ± std)
- HOTA table (per-arch single-seed, multi-seed if reached)
- Ego-Z ablation result
- Why succeeded/failed (mechanism analysis: cross-manifold corruption? feature scale? ego confound?)
- Connection to Exp 39/41 (input-concat / late-concat NEG precedents)
- Lever status table update
- Cost actuals

- [ ] **Step 8.6: Memory entry**

Write either `project_depth_augmented_gmc_positive.md` or `project_depth_augmented_gmc_negative.md`:

```markdown
---
name: Depth-augmented GMC ({POS|NEG})
description: Stage1 17D motion vector with depth + multi-scale dZ {beats|fails to beat} 0.7793 ceiling and {ships ≥2σ on N archs|kills}
type: project
---

2026-05-09 Depth-augmented GMC (Approach B).

**Result:** Stage1 micro AUC mean {value} ± {std} (n=3 seeds) vs stage1 0.7793.
- iKUN HOTA: {value} vs 44.608 ± 0.024 → Δ={value} ({σ count}σ {POS|NEG})
- FH V1: {value} vs 53.716 ± 0.068 → Δ={value}
- FH V2: {value} vs 42.799 ± 0.047 → Δ={value}

**Ego-Z ablation:** with-comp {value} vs without-comp {value} → ego-Z {is|is not} the sub-lever.

**Why:** {mechanism}.

**How to apply:** {when to invoke depth-aug or what to avoid}.

**Files:** retrospective `docs/superpowers/specs/2026-05-09-depth-augmented-gmc-retrospective.md`,
weights `experiments/depth_v1train/seed{0,1,2}/best.pth`,
caches `gmc_link/depth_cache/z_track_*.json`.
```

Update `MEMORY.md` index:

```markdown
- [project_depth_augmented_gmc_{POS|NEG}.md](project_depth_augmented_gmc_{POS|NEG}.md) — 2026-05-09 17D depth-aug GMC {one-line outcome}
```

- [ ] **Step 8.7: Commit**

```bash
cd /home/seanachan/GMC-Link
git add run_build_gmc_cache_depth.py run_depth_eval.sh \
  docs/superpowers/specs/2026-05-09-depth-augmented-gmc-retrospective.md \
  /home/seanachan/.claude/projects/-home-seanachan-GMC-Link/memory/project_depth_augmented_gmc_*.md \
  /home/seanachan/.claude/projects/-home-seanachan-GMC-Link/memory/MEMORY.md \
  experiments/depth_v1train experiments/depth_no_ego_comp depth_eval_results.txt
git commit -m "$(cat <<'EOF'
docs(depth): retrospective + memory entry for 17D depth-augmented GMC

Cross-arch HOTA pool eval at locked Arm A recipes + ego-Z ablation.
Decision documented; lever status updated.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Risk Register (cross-task)

| Risk | Mitigation | Task |
|---|---|---|
| Depth Anything V2 outputs aren't metric meters | Smoke test in Task 1 fails fast → switch to Metric3D-ViT-L | 1 |
| Per-track Z cache too sparse (track_id mismatch between depth driver and gmc cache) | Both use NeuralSORT `predict.txt` track_ids — verified consistent | 2, 3 |
| Identity-init plumbing bug — extra cols leak gradient at step 0 | Bit-exact test in Task 6 | 6 |
| Ego-Z compensation hides depth signal (over-suppresses) | Step 8.4 ablation isolates | 8 |
| Cross-manifold corruption (Exp 39/41 precedent) at 17D scale | Identity-init grows depth contribution only if useful; if corrupted, micro AUC < 0.779, still safer than input-concat | 6, 7 |
| Train-time vs eval-time ego-Z asymmetry (training uses raw dZ; manager applies cohort comp) | Step 8.4 ablation tests both modes; if mismatch matters, retrain with cohort comp at training time | 4, 5, 8 |
| `motion_dim` ckpt incompatibility | Checkpoint metadata stores `motion_dim`; eval scripts branch | 6, 7 |

---

## Cost Summary

| Task | Wall time | GPU |
|---|---|---|
| 1. Depth extractor + smoke | 30 min | 5 min |
| 2. Cache module + 1-seq smoke | 30 min | 5 min |
| 3. Full cache build (3 archs × all seqs) | 4 hr | 4 hr |
| 4. Manager 17D path | 1 hr | — |
| 5. Dataset 17D | 1 hr | — |
| 6. Alignment identity-init + bit-exact | 30 min | — |
| 7. Train 3 seeds | 6 hr | 6 hr |
| 8. HOTA eval + ablation + retrospective | 4 hr | 3 hr |
| **Total** | **~17 hr** | **~18 hr** |

---

## Reference Files

- Spec: `docs/superpowers/specs/2026-05-09-depth-augmented-gmc-design.md`
- 13D motion builder: `gmc_link/manager.py` (`GMCLinkManager.process_frame`, search `spatial_motion = np.array`)
- Aligner: `gmc_link/alignment.py:MotionLanguageAligner.__init__` (already accepts `motion_dim`)
- CLIP cache template: `gmc_link/clip_cache.py`
- GMC cache builder template: `run_build_gmc_cache.py`
- Diag eval: `diagnostics/diag_gt_cosine_distributions.py` (~line 335 for `motion_dim` read, ~line 359 for model construction)
- Per-arch eval scripts (UNCHANGED): `run_ikun_linear_additive.py`, `run_flexhook_phase5_gmc_sweep.py`, `run_flexhook_v2_raw_sweep.py`
- NeuralSORT tracker: `NeuralSORT/{seq}/{car,pedestrian}/predict.txt`
- KITTI images: `/home/seanachan/data/Dataset/refer-kitti-v2/KITTI/training/image_02/{seq}/{frame:06d}.png`
- Multi-seed baselines: `project_ikun_multiseed_positive.md`, `project_flexhook_multiseed.md`
- Cross-manifold corruption precedent: `project_exp39_clip_concat_negative.md`, `project_exp41_late_concat_negative.md`
