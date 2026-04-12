"""
Grid-search α for additive logit fusion on V1 training data (seqs 0005+0013).

Uses pre-collected fusion_train_data_v1.npz with columns:
  [ikun_logit, gmc_prob, is_motion, label, frame_idx]

For motion/stationary expressions (is_motion > 0):
  final_logit = ikun_logit + α * log(gmc_prob / (1 - gmc_prob))
For appearance-only (is_motion == 0):
  final_logit = ikun_logit  (GMC ignored)

Decision boundary: final_logit > 0  (same as iKUN baseline)
"""

import numpy as np
import argparse


def prob_to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compute_f1(pred, label):
    tp = np.sum(pred & label)
    fp = np.sum(pred & ~label)
    fn = np.sum(~pred & label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def main():
    parser = argparse.ArgumentParser(description="Grid-search α for additive logit fusion")
    parser.add_argument("--data", default="gmc_link/fusion_train_data_v1.npz",
                        help="Path to fusion training data")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.025)
    args = parser.parse_args()

    data = np.load(args.data)["samples"]
    ikun_logit = data[:, 0]
    gmc_prob = data[:, 1]
    is_motion = data[:, 2]
    label = data[:, 3].astype(bool)

    gmc_logit = prob_to_logit(gmc_prob)

    # Mask: which samples are motion/stationary (is_motion > 0)
    motion_mask = is_motion > 0

    n_total = len(label)
    n_motion = motion_mask.sum()
    n_appear = n_total - n_motion
    print(f"Samples: {n_total} total, {n_motion} motion/stationary, {n_appear} appearance-only")
    print(f"Positives: {label.sum()} ({label.mean()*100:.1f}%)")

    # Baseline (α=0): all decisions from iKUN alone
    baseline_pred = ikun_logit > 0
    baseline_f1, baseline_p, baseline_r = compute_f1(baseline_pred, label)
    print(f"\nBaseline (α=0.0): F1={baseline_f1:.4f}  P={baseline_p:.4f}  R={baseline_r:.4f}")

    # Grid search
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)

    print(f"\n{'α':>6s}  {'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'ΔF1':>7s}")
    print("-" * 42)

    best_alpha, best_f1 = 0.0, baseline_f1
    for alpha in alphas:
        # For motion/stationary: add α * gmc_logit; for appearance: just ikun_logit
        final_logit = ikun_logit + alpha * gmc_logit * motion_mask
        pred = final_logit > 0
        f1, prec, rec = compute_f1(pred, label)
        delta = f1 - baseline_f1
        marker = " *" if f1 > best_f1 else ""
        print(f"{alpha:6.3f}  {f1:7.4f}  {prec:7.4f}  {rec:7.4f}  {delta:+7.4f}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha

    print(f"\nBest α = {best_alpha:.3f}  (F1={best_f1:.4f}, ΔF1={best_f1 - baseline_f1:+.4f})")

    # Also show motion-only subset performance
    print("\n── Motion/stationary subset only ──")
    motion_label = label[motion_mask]
    motion_ikun = ikun_logit[motion_mask]
    motion_gmc = gmc_logit[motion_mask]

    base_pred_m = motion_ikun > 0
    base_f1_m, base_p_m, base_r_m = compute_f1(base_pred_m, motion_label)
    print(f"Baseline: F1={base_f1_m:.4f}  P={base_p_m:.4f}  R={base_r_m:.4f}")

    best_pred_m = (motion_ikun + best_alpha * motion_gmc) > 0
    best_f1_m, best_p_m, best_r_m = compute_f1(best_pred_m, motion_label)
    print(f"α={best_alpha:.3f}:  F1={best_f1_m:.4f}  P={best_p_m:.4f}  R={best_r_m:.4f}  "
          f"(ΔF1={best_f1_m - base_f1_m:+.4f})")


if __name__ == "__main__":
    main()
