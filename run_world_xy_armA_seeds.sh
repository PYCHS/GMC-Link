#!/usr/bin/env bash
# HOTA Arm A eval on world-XY 17D ckpts (n=3 per arch).
# Compare vs 17D depth-aug shipped: iKUN [44.876, 44.800, 44.793].
set -euo pipefail
cd /home/seanachan/GMC-Link

PY="conda run -n RMOT python"
OUT_DIR=experiments/world_xy_v1train
LOG_IKUN=${OUT_DIR}/ikun_armA_multiseed.log
LOG_V1=${OUT_DIR}/fh_v1_armA_multiseed.log
LOG_V2=${OUT_DIR}/fh_v2_armA_multiseed.log
: > "${LOG_IKUN}"; : > "${LOG_V1}"; : > "${LOG_V2}"

# iKUN locked Arm A: α_m=1 sc_m=0.9 thr_m=0.17 | α_a=1 sc_a=0.30 thr_a=0.10
for seed in 0 1 2; do
  echo "=== iKUN seed=${seed} world-XY Arm A ===" | tee -a "${LOG_IKUN}"
  GMC_SUFFIX=_world_xy_seed${seed} OUT_SUFFIX=_world_xy_seed${seed} \
    ${PY} run_ikun_linear_additive.py \
      --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
      --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
      2>&1 | tee -a "${LOG_IKUN}"
done

# FH V1 locked: α_m=0.65 sc_m=10 thr_m=+3 | α_a=1 sc_a=3.5 thr_a=+0.9
for seed in 0 1 2; do
  echo "=== FH V1 seed=${seed} world-XY ship ===" | tee -a "${LOG_V1}"
  GMC_SUFFIX=_world_xy_seed${seed} OUT_SUFFIX=_world_xy_seed${seed} \
    ${PY} run_flexhook_phase5_gmc_sweep.py \
      --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
      --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
      2>&1 | tee -a "${LOG_V1}"
done

# FH V2 locked: α_m=0.4 sc_m=10 thr_m=+1.3 | α_a=1 sc_a=3.5 thr_a=+1.2
for seed in 0 1 2; do
  echo "=== FH V2 seed=${seed} world-XY ship ===" | tee -a "${LOG_V2}"
  GMC_SUFFIX=_world_xy_seed${seed} OUT_SUFFIX=_world_xy_seed${seed} \
    ${PY} run_flexhook_v2_raw_sweep.py \
      --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
      --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
      2>&1 | tee -a "${LOG_V2}"
done

echo "=== world-XY HOTA done $(date -Iseconds) ==="
grep -E "^\s*pooled=" "${LOG_IKUN}" "${LOG_V1}" "${LOG_V2}"
