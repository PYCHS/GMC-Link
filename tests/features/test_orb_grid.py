"""Unit tests for gmc_link.features.orb_grid.compute_orb_grid_flow."""
from __future__ import annotations

import numpy as np
import pytest

from gmc_link.features.orb_grid import compute_orb_grid_flow


def _make_textured_frame(h: int = 240, w: int = 640, seed: int = 0) -> np.ndarray:
    """Make a deterministic textured grayscale frame ORB can latch onto."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    # Boost local contrast with a tiled pattern so ORB detects everywhere.
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (((xx // 8) + (yy // 8)) % 2).astype(np.uint8) * 80
    return np.clip(base.astype(np.int32) + pattern, 0, 255).astype(np.uint8)


def _shift_frame(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a frame by (dx, dy); pad with zeros."""
    h, w = img.shape
    out = np.zeros_like(img)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_x0 = src_x0 + dx
    dst_y0 = src_y0 + dy
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def test_zero_output_on_identical_frames():
    img = _make_textured_frame(seed=1)
    out = compute_orb_grid_flow(img, img, n_rows=3, n_cols=8)
    assert out.shape == (3 * 8 * 2,)
    assert out.dtype == np.float32
    # Identical frames → all matches have (dx, dy) == (0, 0); per-cell median
    # is exactly zero. Cells without enough matches are also zero.
    assert np.all(out == 0.0)


def test_consistent_translation_recovered():
    img = _make_textured_frame(seed=2)
    dx, dy = 4, -3
    img2 = _shift_frame(img, dx, dy)
    out = compute_orb_grid_flow(img, img2, n_rows=3, n_cols=8)
    assert out.shape == (3 * 8 * 2,)

    # Look at non-zero cells (cells that had enough matches). Their median
    # (dx, dy) should be close to the synthetic shift.
    pairs = out.reshape(-1, 2)
    nonzero_mask = np.any(pairs != 0.0, axis=1)
    assert nonzero_mask.sum() >= 4, (
        f"Expected at least 4 non-empty cells under a uniform translation, got "
        f"{int(nonzero_mask.sum())}"
    )
    nz = pairs[nonzero_mask]
    # Allow ±1 px slack from ORB sub-pixel keypoint location noise.
    assert np.median(nz[:, 0]) == pytest.approx(dx, abs=1.0)
    assert np.median(nz[:, 1]) == pytest.approx(dy, abs=1.0)


def test_empty_cells_zero_filled():
    # Force an extreme grid so most cells will be below min_matches_per_cell.
    img = _make_textured_frame(h=120, w=160, seed=3)
    img2 = _shift_frame(img, 2, 1)
    out = compute_orb_grid_flow(
        img, img2,
        n_rows=10, n_cols=10,
        max_features=80,            # few keypoints
        min_matches_per_cell=20,    # high threshold → most cells empty
    )
    assert out.shape == (10 * 10 * 2,)
    n_empty_cells = int(np.sum(np.all(out.reshape(-1, 2) == 0.0, axis=1)))
    assert n_empty_cells >= 90, (
        f"Expected most cells zero-filled with starved keypoints + high "
        f"threshold, got {n_empty_cells} empty out of 100"
    )


def test_handles_degenerate_inputs():
    # All-zeros frames give ORB nothing to detect.
    blank = np.zeros((240, 640), dtype=np.uint8)
    out = compute_orb_grid_flow(blank, blank, n_rows=3, n_cols=8)
    assert out.shape == (48,)
    assert np.all(out == 0.0)

    # Mismatched shapes — defensive return.
    a = _make_textured_frame(seed=4)
    b = _make_textured_frame(h=200, w=400, seed=5)
    out2 = compute_orb_grid_flow(a, b, n_rows=3, n_cols=8)
    assert out2.shape == (48,)
    assert np.all(out2 == 0.0)
