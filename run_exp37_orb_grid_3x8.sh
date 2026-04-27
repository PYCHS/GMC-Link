#!/usr/bin/env bash
# Exp 37 zoned-orb 3x8 — per-cell ORB-keypoint median flow concat to base 13D.
#
# Motivation:
#   The Farneback per-cell mean variant (zoned_flow_3x8, 48D, 61D total) was NEG
#   (-0.062 micro AUC vs stage1 0.767). Hypothesis: ORB sparse keypoints with
#   Lowe-ratio + outlier rejection give cleaner per-cell motion than Farneback's
#   dense flow on KITTI's textureless asphalt regions.
#
# Cache: cache/orb_grid/3x8/<seq>/<frame:06d>_gap5.npz, populated by
#   `python precompute_orb_grid_3x8.py --all`.

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=/home/seanachan/GMC-Link/diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=/home/seanachan/GMC-Link/diagnostics/results
OUT="${RESULTS_DIR}/exp37"
TAG=v1train_orb_grid_3x8
WEIGHTS=/home/seanachan/GMC-Link/gmc_link_weights_${TAG}.pth

mkdir -p "${OUT}"

# ─── Train ──────────────────────────────────────────────────────────
echo "============================================================"
echo "Training: 13D + zoned_orb_flow_3x8 (61D) → ${WEIGHTS}"
echo "============================================================"
"${PY}" -m gmc_link.train \
  --split v1 \
  --ego orb \
  --extra-features zoned_orb_flow_3x8 \
  --epochs 100 \
  --lr 1e-3 \
  --batch-size 128 \
  --save-path "${WEIGHTS}"

# ─── Eval (held-out V1 seqs) ────────────────────────────────────────
SEQS=(0005 0011 0013)
for seq in "${SEQS[@]}"; do
  echo "--- Diag eval seq=${seq} / ${TAG} ---"
  "${PY}" "${DIAG}" --weights "${WEIGHTS}" --seq "${seq}"
  src_npz="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
  dst_npz="${OUT}/layer3_${seq}_${TAG}.npz"
  mv "${src_npz}" "${dst_npz}"
  echo "  saved: ${dst_npz}"
  src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
  if [[ -f "${src_png}" ]]; then
    mv "${src_png}" "${OUT}/layer3_${seq}_${TAG}.png"
  fi
done

echo "Eval done. Aggregation runs separately via aggregate_orb_grid_3x8_vs_stage1.py"
