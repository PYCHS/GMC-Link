"""Per-track Z time-series cache: round-trip + lookup."""
from __future__ import annotations

from pathlib import Path

from gmc_link.depth_cache import DepthCache, save_depth_cache


def test_round_trip(tmp_path: Path):
    data = {"42": {"100": 12.5, "101": 13.1}, "7": {"100": 5.5}}
    p = tmp_path / "z_track_ikun_0011.json"
    save_depth_cache(data, p)
    cache = DepthCache.load(p)
    assert cache.lookup(track_id=42, frame_id=100) == 12.5
    assert cache.lookup(track_id=7, frame_id=100) == 5.5
    assert cache.lookup(track_id=99, frame_id=100) is None
    assert cache.lookup(track_id=42, frame_id=999) is None


def test_seq_lookup_normalises_str_keys(tmp_path: Path):
    p = tmp_path / "c.json"
    save_depth_cache({"1": {"5": 7.0}}, p)
    cache = DepthCache.load(p)
    assert cache.lookup(1, 5) == 7.0
    assert cache.lookup("1", "5") == 7.0


def test_save_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "deep" / "nested" / "c.json"
    save_depth_cache({"1": {"5": 7.0}}, p)
    assert p.exists()
