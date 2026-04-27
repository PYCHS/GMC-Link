"""Precompute per-cell ORB-keypoint grid flow at 3x8 grid, gap=5.

Walks Refer-KITTI V1 sequences and writes per-frame 48D vectors to
``cache/orb_grid/3x8/<seq>/<frame:06d>_gap5.npz`` with key ``flow``.

This mirrors the Farneback OMF cache structure under ``cache/omf/orb/``,
swapping per-cell mean Farneback for per-cell median ORB-keypoint motion
(Lowe-ratio-filtered, see ``gmc_link.features.orb_grid``).

Design:
  * Frame source path mirrors ``precompute_farneback_eval.py``:
    ``refer-kitti/KITTI/training/image_02/<seq>/<fid:06d>.png``.
  * Gap=5 only (matches existing zoned_flow_3x8 design + dataset loader).
  * For each frame ``f`` we compute the orb-grid between frames ``f`` and
    ``f+5``. Frames where ``f+5`` is missing are skipped.
  * Multiprocessing across frames (8 workers default; cv2 is GIL-friendly).
  * Idempotent: skips frames whose output ``.npz`` already exists.

Usage::

    python precompute_orb_grid_3x8.py --all
    python precompute_orb_grid_3x8.py --seqs 0001 0002 ...
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from gmc_link.features.orb_grid import compute_orb_grid_flow, cell_match_counts


DATA_ROOT = Path("/home/seanachan/GMC-Link/refer-kitti")
OUT_ROOT = Path("/home/seanachan/GMC-Link/cache/orb_grid/3x8")
GAP = 5
N_ROWS, N_COLS = 3, 8

# V1 train + V1 eval seqs that have image_02 frames in DATA_ROOT.
V1_TRAIN_SEQS = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]
V1_EVAL_SEQS = ["0005", "0011", "0013"]
ALL_SEQS = V1_TRAIN_SEQS + V1_EVAL_SEQS


def _process_one(args: Tuple[Path, Path, Path, int, bool]) -> Tuple[str, str]:
    """Worker: compute and write one (seq, frame) entry. Returns (status, path)."""
    img_dir, out_seq, _seq_name, fid, count_cells = args
    out_npz = out_seq / f"{fid:06d}_gap{GAP}.npz"
    if out_npz.exists():
        return ("skip", str(out_npz))

    prev_path = img_dir / f"{fid:06d}.png"
    next_path = img_dir / f"{fid + GAP:06d}.png"
    if not prev_path.exists() or not next_path.exists():
        return ("miss", str(out_npz))

    prev = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(next_path), cv2.IMREAD_GRAYSCALE)
    if prev is None or curr is None:
        return ("miss", str(out_npz))

    flow = compute_orb_grid_flow(
        prev, curr,
        n_rows=N_ROWS, n_cols=N_COLS,
        max_features=1500, lowe_ratio=0.7, min_matches_per_cell=3,
    )

    extras = {"flow": flow.astype(np.float32)}
    if count_cells:
        counts = cell_match_counts(
            prev, curr, n_rows=N_ROWS, n_cols=N_COLS,
            max_features=1500, lowe_ratio=0.7,
        )
        extras["counts"] = counts.astype(np.int32)

    np.savez_compressed(out_npz, **extras)
    return ("ok", str(out_npz))


def _list_frames(img_dir: Path) -> List[int]:
    return sorted(int(p.stem) for p in img_dir.glob("*.png"))


def process_seq(seq: str, n_workers: int, count_cells: bool) -> dict:
    img_dir = DATA_ROOT / "KITTI" / "training" / "image_02" / seq
    if not img_dir.exists():
        print(f"  SKIP {seq}: no frames at {img_dir}", flush=True)
        return {"seq": seq, "ok": 0, "skip": 0, "miss": 0, "elapsed_s": 0.0}
    out_seq = OUT_ROOT / seq
    out_seq.mkdir(parents=True, exist_ok=True)

    frames = _list_frames(img_dir)
    valid_frames = [f for f in frames if (f + GAP) in set(frames)]
    print(
        f"[{seq}] {len(frames)} frames; {len(valid_frames)} with gap={GAP} pair; "
        f"workers={n_workers}",
        flush=True,
    )

    args_iter = [(img_dir, out_seq, seq, fid, count_cells) for fid in valid_frames]

    t0 = time.time()
    n_ok = n_skip = n_miss = 0
    if n_workers <= 1:
        for a in args_iter:
            status, _ = _process_one(a)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_miss += 1
    else:
        with Pool(processes=n_workers) as pool:
            for status, _ in pool.imap_unordered(_process_one, args_iter, chunksize=8):
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_miss += 1
                done = n_ok + n_skip + n_miss
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [{seq}] {done}/{len(valid_frames)} "
                        f"(ok={n_ok}, skip={n_skip}, miss={n_miss}) "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
    elapsed = time.time() - t0
    print(
        f"[{seq}] DONE. wrote {n_ok}, skipped {n_skip} (existing), missing "
        f"{n_miss}, in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "seq": seq,
        "ok": n_ok, "skip": n_skip, "miss": n_miss,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="*", default=None,
                        help="Sequence ids (e.g. 0001 0002). Default: --all required.")
    parser.add_argument("--all", action="store_true",
                        help="Process V1 train+eval seqs.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--count-cells", action="store_true",
                        help="Also save per-cell match counts (for empty-rate diagnostics).")
    args = parser.parse_args()

    if args.all:
        seqs = ALL_SEQS
    elif args.seqs:
        seqs = args.seqs
    else:
        parser.error("Provide either --all or --seqs.")

    print(f"Precomputing orb-grid 3x8 gap={GAP} for {len(seqs)} seqs → {OUT_ROOT}",
          flush=True)
    print(f"  data root: {DATA_ROOT}", flush=True)
    print(f"  workers: {args.workers}", flush=True)
    print(f"  count-cells: {args.count_cells}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    totals = {"ok": 0, "skip": 0, "miss": 0, "elapsed_s": 0.0}
    for seq in seqs:
        res = process_seq(seq, n_workers=args.workers, count_cells=args.count_cells)
        for k in ("ok", "skip", "miss", "elapsed_s"):
            totals[k] += res[k]
    print(
        f"\nALL DONE. wrote={totals['ok']}, skipped={totals['skip']}, "
        f"missing={totals['miss']}, total {totals['elapsed_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
