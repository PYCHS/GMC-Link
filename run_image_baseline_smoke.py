"""Compare image-domain 17D depth-aug magnitudes to world-XY for calibration."""
import numpy as np
from sentence_transformers import SentenceTransformer

from gmc_link.dataset import build_training_data


def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    encoder.eval()
    motion_data, _, _, _ = build_training_data(
        data_root="/home/seanachan/data/Dataset/refer-kitti",
        sequences=["0001"],
        text_encoder=encoder,
        use_depth=True,
        world_xy=False,
        depth_cache_dir="gmc_link/depth_cache",
    )
    arr = np.stack(motion_data)
    motion_part = np.abs(arr[:, :6])
    pcts = np.percentile(motion_part, [50, 95, 99], axis=0)
    print("IMAGE 17D |dx/dy| percentiles:")
    print(f"  50th: {pcts[0]}")
    print(f"  95th: {pcts[1]}")
    print(f"  99th: {pcts[2]}")
    print(f"Mean 95th: {pcts[1].mean():.4f}")


if __name__ == "__main__":
    main()
