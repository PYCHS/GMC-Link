#!/bin/bash
# World-XY Task 6: stage1 train n=3, seeds 0/1/2 sequential.
# Cache built once on seed 0; seeds 1/2 reuse via key (`world_xy=True` in cache key).
set -euo pipefail
cd /home/seanachan/GMC-Link

LOG_DIR=experiments/world_xy_v1train
mkdir -p "$LOG_DIR"

for s in 0 1 2; do
    echo "=== seed $s start $(date -Iseconds) ==="
    conda run -n RMOT python -m gmc_link.train \
        --split v1 --stage 1 \
        --use-depth --world-xy --seed "$s" \
        --identity-init-depth \
        --epochs 100 \
        --save-path "$LOG_DIR/seed${s}/best.pth" \
        2>&1 | tee "$LOG_DIR/seed${s}/train.log"
    echo "=== seed $s done $(date -Iseconds) ==="
done

echo "=== ALL 3 SEEDS DONE $(date -Iseconds) ==="
