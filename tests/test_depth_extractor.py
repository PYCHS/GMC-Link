"""Smoke test: Depth Anything V2 metric-vkitti-large wrapper outputs metric meters."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gmc_link.depth_extractor import DepthExtractor

KITTI_IMG = "/home/seanachan/data/Dataset/refer-kitti-v2/KITTI/training/image_02/0011/000100.png"


@pytest.fixture(scope="module")
def extractor():
    return DepthExtractor(device="cuda")


def test_output_shape_and_dtype(extractor):
    img = np.array(Image.open(KITTI_IMG).convert("RGB"))
    depth = extractor.extract(img)
    assert depth.dtype == np.float32
    assert depth.ndim == 2
    assert depth.shape == img.shape[:2]


def test_metric_range_kitti(extractor):
    img = np.array(Image.open(KITTI_IMG).convert("RGB"))
    depth = extractor.extract(img)
    valid = depth[(depth > 0) & (depth < 200)]
    assert valid.size > 0.5 * depth.size, "too many invalid depths"
    median = float(np.median(valid))
    assert 3.0 < median < 80.0, f"median {median} not KITTI-plausible"


def test_patch_median(extractor):
    img = np.array(Image.open(KITTI_IMG).convert("RGB"))
    depth = extractor.extract(img)
    cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
    patch = depth[cy - 2:cy + 3, cx - 2:cx + 3]
    z = float(np.median(patch))
    assert 0.5 < z < 200.0
