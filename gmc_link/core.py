# gmc_link/core.py
"""
Core utilities for extracting camera ego-motion via ORB features and homography.
"""
from typing import Optional, List, Tuple
import cv2
import numpy as np
from .utils import warp_points
class ORBHomographyEngine:
    """
    Compute rigid background motion (ego-motion) between frames using ORB features
    and RANSAC homography estimation. Masking foreground objects ensures we
    only track the true camera motion.
    """

    def __init__(self, max_features: int = 1500) -> None:
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate_homography(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the 3x3 homography matrix H_prev_to_curr that transforms points
        from prev_frame to curr_frame coordinates.

        Returns:
            (H, bg_residual): H is 3x3 homography; bg_residual is (2,) median
            absolute warp residual of RANSAC inliers in pixels (background noise floor).
        """
        prev_gray = (
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            if len(prev_frame.shape) == 3
            else prev_frame
        )
        curr_gray = (
            cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            if len(curr_frame.shape) == 3
            else curr_frame
        )

        mask = None
        if prev_bboxes:
            h, w = prev_gray.shape
            mask = np.ones((h, w), dtype=np.uint8) * 255
            for bbox in prev_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        kp1, des1 = self.orb.detectAndCompute(prev_gray, mask=mask)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, mask=None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        if len(good_matches) < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        homography_matrix, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if homography_matrix is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        # Compute background residual: median abs warp error of RANSAC inliers
        H = homography_matrix.astype(np.float32)
        if inlier_mask is not None and inlier_mask.sum() > 0:
            inlier_idx = inlier_mask.ravel().astype(bool)
            src_inliers = src_pts[inlier_idx].reshape(-1, 2)
            dst_inliers = dst_pts[inlier_idx].reshape(-1, 2)
            warped_src = warp_points(src_inliers, H)
            residuals = np.abs(dst_inliers - warped_src)
            bg_residual = np.median(residuals, axis=0).astype(np.float32)
        else:
            bg_residual = np.zeros(2, dtype=np.float32)

        return H, bg_residual
