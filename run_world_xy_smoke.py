"""Smoke-build world_xy training cache on a single sequence + report magnitudes.

Output: 50/95/99 percentiles of |dx_s|, |dy_s|, |dx_m|, |dy_m|, |dx_l|, |dy_l|.
Calibration target: 95th-percentile ~1.0. Tune VELOCITY_SCALE_WORLD if outside [0.5, 5.0].
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from gmc_link.dataset import build_training_data


def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    encoder.eval()
    # Single seq smoke for speed
    motion_data, language_data, labels, _id_to_class = build_training_data(
        data_root="/home/seanachan/data/Dataset/refer-kitti",
        sequences=["0001"],
        text_encoder=encoder,
        use_depth=True,
        world_xy=True,
        depth_cache_dir="gmc_link/depth_cache",
    )
    arr = np.stack(motion_data)
    print(f"Built {arr.shape[0]} samples, dim={arr.shape[1]}")
    print()

    # First 6 dims are world dx/dy across 3 scales
    motion_part = np.abs(arr[:, :6])
    pcts = np.percentile(motion_part, [50, 95, 99], axis=0)
    print("World |dx/dy| percentiles (slot order: dx_s, dy_s, dx_m, dy_m, dx_l, dy_l):")
    print(f"  50th: {pcts[0]}")
    print(f"  95th: {pcts[1]}")
    print(f"  99th: {pcts[2]}")
    print()
    print(f"Mean 95th across 6 slots: {pcts[1].mean():.4f}")
    print(f"Calibration target: 1.0 (acceptable [0.5, 5.0])")

    # Sanity: depth slots
    if arr.shape[1] >= 17:
        z_n = arr[:, 13]
        print(f"Z_n (slot 13) min/median/max: {z_n.min():.3f}/{np.median(z_n):.3f}/{z_n.max():.3f}")

    # NaN check
    n_nan = np.isnan(arr).sum()
    print(f"NaN count: {n_nan}")


if __name__ == "__main__":
    main()
