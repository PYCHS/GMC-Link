#!/usr/bin/env bash
# Patch: build missing iKUN 0011 caches for world-XY (default builder skips 0011).
set -euo pipefail
cd /home/seanachan/GMC-Link

PY="conda run -n RMOT python"
LOG=experiments/world_xy_v1train/cache_build_0011_patch.log
: > "${LOG}"

for s in 0 1 2; do
  CKPT="experiments/world_xy_v1train/seed${s}/best.pth"
  SUFFIX="_world_xy_seed${s}"
  echo "=== iKUN 0011 seed=${s} $(date -Iseconds) ===" | tee -a "${LOG}"
  GMC_WEIGHTS="${CKPT}" GMC_SUFFIX="${SUFFIX}" \
    ${PY} run_build_gmc_cache.py 0011 2>&1 | tee -a "${LOG}"
done

echo "=== iKUN 0011 patch done $(date -Iseconds) ===" | tee -a "${LOG}"
