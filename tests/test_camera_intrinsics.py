"""Unit tests for KITTI camera intrinsics loader."""
from gmc_link.camera_intrinsics import CameraIntrinsics


def test_canonical_kitti():
    intr = CameraIntrinsics()
    fx, fy, cx, cy = intr.get("0005")
    assert abs(fx - 721.5377) < 1e-3
    assert abs(fy - 721.5377) < 1e-3
    assert abs(cx - 609.5593) < 1e-3
    assert abs(cy - 172.8540) < 1e-3


def test_unknown_seq_falls_back_to_canonical():
    intr = CameraIntrinsics()
    fx, fy, cx, cy = intr.get("9999")
    assert fx == 721.5377
    assert fy == 721.5377


def test_overrides_take_precedence():
    custom = {"0005": {"f_x": 700.0, "f_y": 710.0, "c_x": 600.0, "c_y": 170.0}}
    intr = CameraIntrinsics(calib_overrides=custom)
    fx, fy, cx, cy = intr.get("0005")
    assert fx == 700.0
    assert fy == 710.0
    assert cx == 600.0
    assert cy == 170.0
    fx2, _, _, _ = intr.get("0011")
    assert fx2 == 721.5377
