#!/usr/bin/env bash
# HOTA Arm A eval on Exp 39 (CLIP-visual concat 128D) caches at locked recipes.
# Single-seed reference depth-aug: iKUN 44.876 / FH V1 53.787 / FH V2 42.836.
set -euo pipefail
cd /home/seanachan/GMC-Link

PY="conda run -n RMOT python"
OUT_DIR=experiments/exp39_clip128
LOG_IKUN=${OUT_DIR}/ikun_armA_hota.log
LOG_V1=${OUT_DIR}/fh_v1_armA_hota.log
LOG_V2=${OUT_DIR}/fh_v2_armA_hota.log
: > "${LOG_IKUN}"; : > "${LOG_V1}"; : > "${LOG_V2}"

SUFFIX=_exp39_clip128

echo "=== iKUN Arm A Exp39 ===" | tee -a "${LOG_IKUN}"
GMC_SUFFIX=${SUFFIX} OUT_SUFFIX=${SUFFIX} \
  ${PY} run_ikun_linear_additive.py \
    --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
    --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
    2>&1 | tee -a "${LOG_IKUN}"

echo "=== FH V1 Arm A Exp39 ===" | tee -a "${LOG_V1}"
GMC_SUFFIX=${SUFFIX} OUT_SUFFIX=${SUFFIX} \
  ${PY} run_flexhook_phase5_gmc_sweep.py \
    --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
    --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
    2>&1 | tee -a "${LOG_V1}"

echo "=== FH V2 Arm A Exp39 ===" | tee -a "${LOG_V2}"
GMC_SUFFIX=${SUFFIX} OUT_SUFFIX=${SUFFIX} \
  ${PY} run_flexhook_v2_raw_sweep.py \
    --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
    --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
    2>&1 | tee -a "${LOG_V2}"

echo "=== Exp 39 HOTA done $(date -Iseconds) ==="
grep -E "^\s*pooled=" "${LOG_IKUN}" "${LOG_V1}" "${LOG_V2}"
