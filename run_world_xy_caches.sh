#!/usr/bin/env bash
# Build 9 GMC caches (3 archs × 3 seeds) for world-XY 17D ckpts.
# Caches: gmc_link/gmc_scores_{arch}_{seq}_world_xy_seed{s}_cache.json
set -euo pipefail
cd /home/seanachan/GMC-Link

PY="conda run -n RMOT python"
LOG_DIR=experiments/world_xy_v1train
LOG="${LOG_DIR}/cache_build.log"
: > "${LOG}"

for s in 0 1 2; do
  CKPT="experiments/world_xy_v1train/seed${s}/best.pth"
  SUFFIX="_world_xy_seed${s}"
  for builder in run_build_gmc_cache.py run_build_gmc_cache_flexhook.py run_build_gmc_cache_flexhook_v2_raw.py; do
    echo "=== seed=${s} builder=${builder} $(date -Iseconds) ===" | tee -a "${LOG}"
    GMC_WEIGHTS="${CKPT}" GMC_SUFFIX="${SUFFIX}" \
      ${PY} "${builder}" 2>&1 | tee -a "${LOG}"
  done
done

echo "=== world-XY cache build done $(date -Iseconds) ===" | tee -a "${LOG}"
