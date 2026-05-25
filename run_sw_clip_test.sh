#!/bin/bash
# Single-seed gate: shared_weight + CLIP early-concat (clip128) on iKUN.
# Anchor = noclip sw seed0: pooled=44.561 APPEAR=46.819 MOVING=28.885 STATIC=43.24
set -euo pipefail
cd /home/seanachan/GMC-Link

W=gmc_link_weights_v1train_sharedweight_clip128_seed0.pth
SUF=_sharedweight_clip128_seed0_rawcos

echo "=== [1/3] TRAIN shared_weight+clip128 seed0 $(date) ==="
python -m gmc_link.train --split v1 --stage 1 --architecture shared_weight \
  --use-clip-feat --clip-dim 128 --seed 0 --save-path "$W"

echo "=== [2/3] BUILD iKUN GMC caches raw-cos $(date) ==="
for seq in 0005 0011 0013; do
  GMC_WEIGHTS="$W" GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_build_gmc_cache.py "$seq"
done

echo "=== [3/3] EVAL iKUN clip-sw-seed0 (sw ship recipe) $(date) ==="
GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_ikun_linear_additive.py \
  --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
  --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10

echo "=== ANCHOR noclip-sw-seed0: pooled=44.561 APPEAR=46.819 MOVING=28.885 STATIC=43.24 ==="
echo "=== DONE $(date) ==="
