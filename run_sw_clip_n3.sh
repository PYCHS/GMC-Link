#!/bin/bash
# shared-weight + CLIP early-concat (clip128) → n=3. Train seeds 1+2 (seed0 done),
# build iKUN/FH V1/FH V2 caches, eval all 3 archs × 3 seeds.
set -euo pipefail
cd /home/seanachan/GMC-Link

for s in 1 2; do
  W=gmc_link_weights_v1train_sharedweight_clip128_seed${s}.pth
  SUF=_sharedweight_clip128_seed${s}_rawcos
  echo "=== TRAIN seed$s $(date) ==="
  python -m gmc_link.train --split v1 --stage 1 --architecture shared_weight \
    --use-clip-feat --clip-dim 128 --seed $s --save-path "$W"
  echo "=== iKUN caches seed$s $(date) ==="
  for seq in 0005 0011 0013; do
    GMC_WEIGHTS="$W" GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_build_gmc_cache.py "$seq"
  done
  echo "=== FH V1 caches seed$s $(date) ==="
  for seq in 0005 0011 0013; do
    GMC_WEIGHTS="$W" GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_build_gmc_cache_flexhook.py "$seq"
  done
  echo "=== FH V2 caches seed$s $(date) ==="
  for seq in 0005 0011 0013 0019; do
    GMC_WEIGHTS="$W" GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_build_gmc_cache_flexhook_v2_raw.py "$seq"
  done
done

echo "=== EVAL n=3 (all archs, seeds 0/1/2) $(date) ==="
for s in 0 1 2; do
  SUF=_sharedweight_clip128_seed${s}_rawcos
  echo "--- iKUN seed$s ---"
  GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_ikun_linear_additive.py \
    --alpha 1.0 --gmc_scale 0.9 --thr 0.17 --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 2>&1 | grep -iE "pooled=" | tail -1
  echo "--- FHV1 seed$s ---"
  GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_flexhook_phase5_gmc_sweep.py \
    --alpha 0.65 --gmc_scale 10.0 --thr 3.0 --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 2>&1 | grep -iE "pooled=" | tail -1
  echo "--- FHV2 seed$s ---"
  GMC_SUFFIX="$SUF" GMC_RAW_COS=1 python run_flexhook_v2_raw_sweep.py \
    --alpha 0.4 --gmc_scale 10.0 --thr 1.3 --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 2>&1 | grep -iE "pooled=" | tail -1
done
echo "=== DONE $(date) ==="
