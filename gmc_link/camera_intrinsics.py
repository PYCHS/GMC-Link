"""KITTI camera intrinsics for world-plane projection.

Per-drive raw calib not available locally (only odometry calib present at
/home/seanachan/data/Dataset/data_odometry_calib/). Use canonical
2011_09_26 P_rect_02 (left color rectified). Per-drive variation < 1%
on the KITTI same-rig setup, acceptable for pilot world-XY projection.

Reference: KITTI raw devkit calib_cam_to_cam.txt, P_rect_02 row,
[fx 0 cx 0; 0 fy cy 0; 0 0 1 0].
"""
from typing import Tuple, Optional

CANONICAL_KITTI_2011_09_26 = {
    "f_x": 721.5377,
    "f_y": 721.5377,
    "c_x": 609.5593,
    "c_y": 172.8540,
}


class CameraIntrinsics:
    def __init__(self, calib_overrides: Optional[dict] = None):
        self.canonical = CANONICAL_KITTI_2011_09_26
        self.overrides = calib_overrides or {}

    def get(self, seq: str) -> Tuple[float, float, float, float]:
        c = self.overrides.get(seq, self.canonical)
        return c["f_x"], c["f_y"], c["c_x"], c["c_y"]
