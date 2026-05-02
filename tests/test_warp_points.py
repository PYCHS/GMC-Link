"""Tests for ``warp_points`` numerical robustness.

``warp_points`` sits on the hot path of both the per-frame ego-residual
computation in ``core.py`` (RANSAC inlier residual) and the cumulative-homography
centroid warp in ``manager.py``. Cumulative composition over ``frame_gap=10``
can drift the bottom row of H enough to send a centroid onto the homography's
line at infinity (w → 0), historically producing ``inf``/``nan`` in the 13D
motion vector and poisoning the aligner's LayerNorm outputs.

These tests pin the public contract: identity, translation, empty input, and
two degeneracy cases that previously returned non-finite values.
"""
import numpy as np

from gmc_link.utils import warp_points


def test_warp_points_identity_homography_returns_input():
    H = np.eye(3, dtype=np.float32)
    pts = np.array([[10.0, 20.0], [-5.0, 7.5], [0.0, 0.0]], dtype=np.float32)
    out = warp_points(pts, H)
    np.testing.assert_allclose(out, pts, atol=1e-6)


def test_warp_points_pure_translation_shifts_correctly():
    H = np.array([[1, 0, 3], [0, 1, -4], [0, 0, 1]], dtype=np.float32)
    pts = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    expected = pts + np.array([3.0, -4.0])
    np.testing.assert_allclose(warp_points(pts, H), expected, atol=1e-6)


def test_warp_points_empty_input_returns_empty_float_array():
    H = np.eye(3, dtype=np.float32)
    out = warp_points(np.empty((0, 2)), H)
    assert out.shape == (0, 2)
    assert np.issubdtype(out.dtype, np.floating)


def test_warp_points_degenerate_homography_stays_finite():
    """Bottom row of H is all-zero so the divisor is exactly 0 for every point.
    Pre-fix this returned NaN/Inf and silently corrupted the 13D motion vector."""
    H = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
    pts = np.array([[0.0, 0.0], [50.0, -25.0]], dtype=np.float32)
    out = warp_points(pts, H)
    assert np.isfinite(out).all(), f"warp_points returned non-finite values: {out}"


def test_warp_points_near_zero_w_does_not_explode():
    """A perspective H with a small nonzero bottom row can still drive w to ~0
    for a point far from the origin (this is the realistic cumulative-drift case
    that the manager's frame_gap=10 composition can reach)."""
    # divisor for point (-1000, 0) is 0.001 * (-1000) + 1 = 0
    H = np.array([[1, 0, 0], [0, 1, 0], [0.001, 0, 1]], dtype=np.float32)
    pts = np.array([[-1000.0, 0.0]], dtype=np.float32)
    out = warp_points(pts, H)
    assert np.isfinite(out).all(), f"warp_points returned non-finite values: {out}"
