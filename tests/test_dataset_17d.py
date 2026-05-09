"""build_training_data emits 17D when use_depth=True."""
from __future__ import annotations

from pathlib import Path

from gmc_link.dataset import build_training_data
from gmc_link.text_utils import TextEncoder


def _encoder():
    return TextEncoder(model_name="all-MiniLM-L6-v2", device="cpu")


def test_use_depth_kwarg_accepted():
    """Smoke: signature accepts use_depth + depth_cache_dir kwargs."""
    import inspect
    sig = inspect.signature(build_training_data)
    assert "use_depth" in sig.parameters
    assert "depth_cache_dir" in sig.parameters


def test_13d_default_no_depth(tmp_path: Path):
    encoder = _encoder()
    motion_data, *_ = build_training_data(
        data_root="refer-kitti",
        sequences=["0001"],
        text_encoder=encoder,
    )
    assert len(motion_data) > 0
    assert motion_data[0].shape[0] == 13


def test_17d_with_depth(tmp_path: Path):
    encoder = _encoder()
    cache_dir = Path("/home/seanachan/GMC-Link/gmc_link/depth_cache")
    motion_data, *_ = build_training_data(
        data_root="refer-kitti",
        sequences=["0001"],
        text_encoder=encoder,
        use_depth=True,
        depth_cache_dir=str(cache_dir),
    )
    assert len(motion_data) > 0
    assert motion_data[0].shape[0] == 17
