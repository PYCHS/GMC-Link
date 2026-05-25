#!/bin/bash
# shared-weight + CLIP early-concat (clip128 seed0) on FlexHook V1 + V2.
# Reuses the V1-trained aligner already built for the iKUN test.
set -euo pipefail
cd /home/seanachan/GMC-Link

W=gmc_link_weights_v1train_sharedweight_clip128_seed0.pth
SUF=_sharedweight_clip128_seed0_rawcos

echo "=== [1/4] BUILD FH V1 clip caches $(date) ==="
for seq in 0005 0011 0013; do
  GMC_WEIGHTS="$W" GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_build_gmc_cache_flexhook.py "$seq"
done

echo "=== [2/4] BUILD FH V2 clip caches $(date) ==="
for seq in 0005 0011 0013 0019; do
  GMC_WEIGHTS="$W" GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_build_gmc_cache_flexhook_v2_raw.py "$seq"
done

echo "=== [3/4] EVAL FH V1 clip (sw recipe) $(date) ==="
GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_flexhook_phase5_gmc_sweep.py \
  --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
  --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9

echo "=== [4/4] EVAL FH V2 clip (sw recipe) $(date) ==="
GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_flexhook_v2_raw_sweep.py \
  --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
  --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2

echo "=== DONE $(date) ==="
