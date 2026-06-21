#!/usr/bin/env bash
# Build GMC caches with Exp 39 clip128 ckpt (CLIP-visual concat into 13D motion).
# Single-seed HOTA revisit — depth-aug precedent: AUC NEG can survive HOTA.
set -euo pipefail
cd /home/seanachan/GMC-Link

PY="conda run -n RMOT python"
CKPT="experiments/exp39_clip128/weights.pth"
SUFFIX="_exp39_clip128"
LOG_DIR=experiments/exp39_clip128
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/cache_build.log"
: > "${LOG}"

echo "=== iKUN seqs 0005 0011 0013 $(date -Iseconds) ===" | tee -a "${LOG}"
GMC_WEIGHTS="${CKPT}" GMC_SUFFIX="${SUFFIX}" \
  ${PY} run_build_gmc_cache.py 0005 0011 0013 2>&1 | tee -a "${LOG}"

echo "=== FH V1 $(date -Iseconds) ===" | tee -a "${LOG}"
GMC_WEIGHTS="${CKPT}" GMC_SUFFIX="${SUFFIX}" \
  ${PY} run_build_gmc_cache_flexhook.py 2>&1 | tee -a "${LOG}"

echo "=== FH V2 $(date -Iseconds) ===" | tee -a "${LOG}"
GMC_WEIGHTS="${CKPT}" GMC_SUFFIX="${SUFFIX}" \
  ${PY} run_build_gmc_cache_flexhook_v2_raw.py 2>&1 | tee -a "${LOG}"

echo "=== exp39 cache build done $(date -Iseconds) ===" | tee -a "${LOG}"
ls -la gmc_link/gmc_scores_*${SUFFIX}_cache.json | tee -a "${LOG}"
