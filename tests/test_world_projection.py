"""Unit tests for world-XY projection in GMCLinkManager."""
import numpy as np
import pytest

from gmc_link.manager import GMCLinkManager


def test_world_xy_flag_constructor():
    """world_xy kwarg accepted, defaults False, persists."""
    mgr_img = GMCLinkManager(weights_path=None, device="cpu", world_xy=False)
    mgr_world = GMCLinkManager(weights_path=None, device="cpu", world_xy=True)
    assert mgr_img.world_xy is False
    assert mgr_world.world_xy is True


def test_world_xy_intrinsics_loaded():
    """world_xy=True instantiates CameraIntrinsics."""
    mgr_world = GMCLinkManager(weights_path=None, device="cpu", world_xy=True)
    assert mgr_world.intrinsics is not None
    fx, fy, cx, cy = mgr_world.intrinsics.get("0005")
    assert abs(fx - 721.5377) < 1e-3


def test_world_xy_off_no_intrinsics():
    """world_xy=False leaves intrinsics None."""
    mgr_img = GMCLinkManager(weights_path=None, device="cpu", world_xy=False)
    assert mgr_img.intrinsics is None


def test_velocity_scale_world_constant():
    """VELOCITY_SCALE_WORLD class constant exposed for ckpt-meta persistence."""
    assert hasattr(GMCLinkManager, "VELOCITY_SCALE_WORLD")
    assert GMCLinkManager.VELOCITY_SCALE_WORLD == 2.0


def test_world_projection_math():
    """Inverse pinhole: world_dX = pixel_dx * Z / f_x.

    For dx=10 px, Z=30 m, f_x=721.5377 → world_dX ≈ 0.4158 m.
    """
    pixel_dx = 10.0
    z = 30.0
    f_x = 721.5377
    expected_world_m = pixel_dx * z / f_x
    assert abs(expected_world_m - 0.4158) < 1e-3
