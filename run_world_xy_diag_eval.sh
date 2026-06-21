#!/bin/bash
# Task 6 step 3: diag eval n=3 × 3 V1 test seqs.
# Output: layer3_{seq}_v1train_world_xy_seed{0,1,2}.npz (aggregator pattern)
set -euo pipefail
cd /home/seanachan/GMC-Link

RESULTS=diagnostics/results
SEQS=(0005 0011 0013)
SEEDS=(0 1 2)

for s in "${SEEDS[@]}"; do
    for q in "${SEQS[@]}"; do
        TAG="v1train_world_xy_seed${s}"
        OUT="${RESULTS}/layer3_${q}_${TAG}.npz"
        if [[ -f "$OUT" ]]; then
            echo "[skip] $OUT exists"
            continue
        fi
        echo "=== seed $s seq $q $(date -Iseconds) ==="
        conda run -n RMOT python diagnostics/diag_gt_cosine_distributions.py \
            --weights "experiments/world_xy_v1train/seed${s}/best.pth" \
            --seq "$q"
        # Rename default npz → tagged
        mv "${RESULTS}/layer3_gt_cosine_${q}.npz" "$OUT"
        # Optional: mv png too if present
        [[ -f "${RESULTS}/layer3_gt_cosine_${q}.png" ]] && \
            mv "${RESULTS}/layer3_gt_cosine_${q}.png" "${RESULTS}/layer3_${q}_${TAG}.png" || true
    done
done

echo "=== diag eval done $(date -Iseconds) ==="
