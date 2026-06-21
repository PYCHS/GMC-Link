"""Paired t-test: 17D world-XY vs 17D depth-aug at locked Arm A recipes."""
import numpy as np
from scipy import stats

# n=3 pooled HOTA per arch
data = {
    "iKUN":  {"depth": [44.876, 44.800, 44.793], "world": [44.909, 44.860, 44.701]},
    "FH V1": {"depth": [53.787, 53.809, 53.698], "world": [53.610, 53.625, 53.742]},
    "FH V2": {"depth": [42.836, 42.836, 42.828], "world": [42.778, 42.906, 42.817]},
}

print(f"{'arch':<6} {'depth_mean':>10} {'world_mean':>11} {'delta':>8} {'t':>7} {'p_two':>7} {'p_one_pos':>10}")
print("-" * 70)
for arch, d in data.items():
    depth = np.array(d["depth"])
    world = np.array(d["world"])
    delta = world - depth
    t, p_two = stats.ttest_rel(world, depth)
    p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
    print(f"{arch:<6} {depth.mean():>10.3f} {world.mean():>11.3f} {delta.mean():>+8.3f} {t:>+7.3f} {p_two:>7.4f} {p_one:>10.4f}")

print("\n=== Decision ===")
print("Reference depth-aug shipped: iKUN +0.215 sig p=0.016, FH V1 +0.048, FH V2 +0.034 within seed noise")
print("Per spec §5.5 ship gate: HOTA POS at any arch + paper > paper-only baseline")
