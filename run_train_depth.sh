#!/bin/bash
set -e
cd /home/seanachan/GMC-Link
for SEED in 0 1 2; do
  echo "=== seed $SEED ==="
  HF_HUB_OFFLINE=1 conda run -n RMOT python -m gmc_link.train \
    --split v1 \
    --use-depth \
    --identity-init-depth \
    --depth-cache-dir gmc_link/depth_cache \
    --seed $SEED \
    --epochs 100 --batch-size 256 \
    --save-path experiments/depth_v1train/seed${SEED}/best.pth \
    2>&1 | tee experiments/depth_v1train/seed${SEED}/train.log
done
echo "=== done"
