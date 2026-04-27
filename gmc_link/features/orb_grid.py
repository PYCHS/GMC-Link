"""Per-cell ORB-keypoint grid flow (Exp 37 zoned-orb variant).

Builds a per-cell 2D motion estimate by ORB-keypoint matching between two
grayscale frames, binning the matches by previous-frame keypoint location into
an n_rows x n_cols grid, and taking the per-cell median (dx, dy).

Hypothesis: ORB sparse keypoints with Lowe-ratio + RANSAC-friendly outlier
rejection give cleaner per-cell motion than Farneback's dense flow on KITTI's
textureless asphalt regions.

Output layout (flat float32 of length n_rows*n_cols*2): row-major cells, each
emitting [median_dx, median_dy]; cells with fewer than ``min_matches_per_cell``
matches are zero-filled.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_orb_grid_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    n_rows: int = 3,
    n_cols: int = 8,
    max_features: int = 1500,
    lowe_ratio: float = 0.7,
    min_matches_per_cell: int = 3,
) -> np.ndarray:
    """Compute per-cell median ORB motion vector between two grayscale frames.

    Parameters
    ----------
    prev_gray, curr_gray : np.ndarray
        Grayscale frames of identical shape (H, W).
    n_rows, n_cols : int
        Grid resolution. Cells are binned by prev-frame keypoint position.
    max_features : int
        Max ORB features per frame.
    lowe_ratio : float
        Lowe's ratio test threshold. Lower = stricter.
    min_matches_per_cell : int
        Minimum number of matches in a cell for it to emit a non-zero estimate.
        Cells below threshold are zero-filled.

    Returns
    -------
    np.ndarray
        Flat float32 vector of length n_rows*n_cols*2. Row-major cell layout,
        per-cell entries are [median_dx, median_dy].
    """
    n_dims = n_rows * n_cols * 2
    out = np.zeros(n_dims, dtype=np.float32)

    if prev_gray is None or curr_gray is None:
        return out
    if prev_gray.ndim != 2 or curr_gray.ndim != 2:
        return out
    H, W = prev_gray.shape[:2]
    if H == 0 or W == 0:
        return out
    if curr_gray.shape[:2] != (H, W):
        return out

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return out

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return out

    pairs = []  # (px, py, dx, dy)
    for mp in knn:
        if len(mp) == 2:
            m, n = mp
            if m.distance < lowe_ratio * n.distance:
                px, py = kp1[m.queryIdx].pt
                qx, qy = kp2[m.trainIdx].pt
                pairs.append((px, py, qx - px, qy - py))
        elif len(mp) == 1:
            m = mp[0]
            px, py = kp1[m.queryIdx].pt
            qx, qy = kp2[m.trainIdx].pt
            pairs.append((px, py, qx - px, qy - py))

    if not pairs:
        return out

    arr = np.asarray(pairs, dtype=np.float32)  # (N, 4)

    # Bin by prev_kp position into cells.
    # We want row r ∈ [r*H/n_rows, (r+1)*H/n_rows). searchsorted with the
    # interior edges (exclusive of 0 and H) gives that exact partition.
    row_edges = np.linspace(0, H, n_rows + 1)
    col_edges = np.linspace(0, W, n_cols + 1)
    rows = np.clip(np.searchsorted(row_edges[1:-1], arr[:, 1]), 0, n_rows - 1)
    cols = np.clip(np.searchsorted(col_edges[1:-1], arr[:, 0]), 0, n_cols - 1)
    cell_ids = rows * n_cols + cols

    for cid in range(n_rows * n_cols):
        sel = arr[cell_ids == cid]
        if sel.shape[0] >= min_matches_per_cell:
            out[cid * 2] = float(np.median(sel[:, 2]))
            out[cid * 2 + 1] = float(np.median(sel[:, 3]))
    return out


def cell_match_counts(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    n_rows: int = 3,
    n_cols: int = 8,
    max_features: int = 1500,
    lowe_ratio: float = 0.7,
) -> np.ndarray:
    """Return per-cell ORB match count (n_rows*n_cols int array).

    Diagnostic helper used to measure the empty-cell rate on real frames; not on
    the training hot path.
    """
    counts = np.zeros(n_rows * n_cols, dtype=np.int32)
    if prev_gray is None or curr_gray is None:
        return counts
    if prev_gray.ndim != 2 or curr_gray.ndim != 2:
        return counts
    H, W = prev_gray.shape[:2]
    if H == 0 or W == 0:
        return counts
    if curr_gray.shape[:2] != (H, W):
        return counts

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return counts
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return counts
    pts = []
    for mp in knn:
        if len(mp) == 2:
            m, n = mp
            if m.distance < lowe_ratio * n.distance:
                px, py = kp1[m.queryIdx].pt
                pts.append((px, py))
        elif len(mp) == 1:
            m = mp[0]
            px, py = kp1[m.queryIdx].pt
            pts.append((px, py))
    if not pts:
        return counts
    arr = np.asarray(pts, dtype=np.float32)
    row_edges = np.linspace(0, H, n_rows + 1)
    col_edges = np.linspace(0, W, n_cols + 1)
    rows = np.clip(np.searchsorted(row_edges[1:-1], arr[:, 1]), 0, n_rows - 1)
    cols = np.clip(np.searchsorted(col_edges[1:-1], arr[:, 0]), 0, n_cols - 1)
    cell_ids = rows * n_cols + cols
    for cid in range(n_rows * n_cols):
        counts[cid] = int(np.sum(cell_ids == cid))
    return counts
