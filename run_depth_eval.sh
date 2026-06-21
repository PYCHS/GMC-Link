#!/usr/bin/env bash
# Eval 17D depth-aug aligner on V1 test (3 seqs × 3 seeds), aggregate pool.
set -euo pipefail
cd /home/seanachan/GMC-Link

PY="conda run -n RMOT python"
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
OUT="${RESULTS_DIR}/depth_v1train"
mkdir -p "${OUT}"

SEQS=(0005 0011 0013)

for SEED in 0 1 2; do
  TAG="depth_seed${SEED}"
  W="experiments/depth_v1train/seed${SEED}/best.pth"
  for SEQ in "${SEQS[@]}"; do
    echo "=== eval ${TAG} seq=${SEQ}"
    HF_HUB_OFFLINE=1 ${PY} "${DIAG}" --weights "${W}" --seq "${SEQ}" \
      2>&1 | tee "${OUT}/diag_${TAG}_${SEQ}.log"
    src="${RESULTS_DIR}/layer3_gt_cosine_${SEQ}.npz"
    dst="${OUT}/layer3_${SEQ}_${TAG}.npz"
    mv "${src}" "${dst}"
    [[ -f "${RESULTS_DIR}/layer3_gt_cosine_${SEQ}.png" ]] && \
      mv "${RESULTS_DIR}/layer3_gt_cosine_${SEQ}.png" "${OUT}/layer3_${SEQ}_${TAG}.png"
  done
done

echo "=== aggregate"
${PY} diagnostics/aggregate_multiseq.py \
  --results-dir "${OUT}" \
  --output-dir "${OUT}/multiseq" \
  --weights depth_seed0 depth_seed1 depth_seed2 \
  --seqs "${SEQS[@]}"

echo "=== done. see ${OUT}/multiseq/"
