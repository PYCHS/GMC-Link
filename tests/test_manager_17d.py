"""GMCLinkManager 17D depth path: motion_dim + ego-Z residual."""
from __future__ import annotations

from gmc_link.manager import GMCLinkManager


def test_13d_default():
    m = GMCLinkManager()
    assert m.motion_dim == 13
    assert m.use_depth is False


def test_17d_with_depth_lookup():
    m = GMCLinkManager(use_depth=True)
    assert m.motion_dim == 17
    assert m.use_depth is True


def test_dz_residual_with_ego_compensation():
    """dZ_residual = dZ_track − median(dZ over stationary tracks).

    All 3 closer by 1m (uniform ego forward) → ego comp removes shift, residual=0.
    """
    m = GMCLinkManager(use_depth=True)
    z_now = {1: 20.0, 2: 30.0, 3: 40.0}
    z_prev = {1: 19.0, 2: 29.0, 3: 39.0}
    stationary_ids = {1, 2, 3}
    dz = m._compute_dz_residual(z_now, z_prev, stationary_ids)
    for tid in (1, 2, 3):
        assert abs(dz[tid] - 0.0) < 1e-5


def test_dz_residual_isolates_approaching_object():
    """tid=3 was 39, now 35 = approaching by 4m vs ego +1m → residual −5m."""
    m = GMCLinkManager(use_depth=True)
    z_now = {1: 20.0, 2: 30.0, 3: 35.0}
    z_prev = {1: 19.0, 2: 29.0, 3: 39.0}
    stationary_ids = {1, 2}
    dz = m._compute_dz_residual(z_now, z_prev, stationary_ids)
    assert abs(dz[1] - 0.0) < 1e-5
    assert abs(dz[2] - 0.0) < 1e-5
    assert abs(dz[3] - (-5.0)) < 1e-5


def test_dz_residual_no_stationary_falls_back_to_zero_ego():
    m = GMCLinkManager(use_depth=True)
    z_now = {1: 20.0}
    z_prev = {1: 25.0}
    dz = m._compute_dz_residual(z_now, z_prev, stationary_ids=set())
    assert abs(dz[1] - (-5.0)) < 1e-5
