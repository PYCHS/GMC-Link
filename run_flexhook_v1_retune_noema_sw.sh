#!/bin/bash
# V1 motion-axis retune on sw+no-EMA. Appear axis locked (1.0, 3.5, +0.9).
# Phase 1: single-seed (seed=0) coarse sweep. Phase 2 (manual): 3-seed validate top-1.
set -e
cd /home/seanachan/GMC-Link

OUT=results/flexhook_v1_retune_noema_sw_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "alpha\tsc\tthr\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

suffix="_sharedweight_seed0_noema_rawcos"

# Coarse sweep: vary motion (α, sc, thr); keep appear ship locked.
# Tag pattern: a{α}_scale{sc}_thr{thr}_aa1.0_sca3.5_thra0.9
declare -a CONFIGS=(
    "0.65 10.0 3.0"   # control (current ship)
    "0.65 10.0 2.5"   # lower thr
    "0.65 10.0 2.0"   # lower thr more
    "0.65 8.0  2.5"   # lower sc + thr
    "0.65 8.0  3.0"   # lower sc only
    "0.5  10.0 2.5"   # lower α + thr
    "0.5  10.0 3.0"   # lower α only
    "0.5  8.0  2.5"   # all gentler
)

for cfg in "${CONFIGS[@]}"; do
    read -r alpha sc thr <<< "$cfg"
    tag="a${alpha}_scale${sc}_thr${thr}_aa1.0_sca3.5_thra0.9"
    log=/tmp/fh_v1_retune_${alpha}_${sc}_${thr}.log
    echo "▶ α=${alpha} sc=${sc} thr=${thr}"
    GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
        python run_flexhook_phase5_gmc_sweep.py \
            --alpha $alpha --gmc_scale $sc --thr $thr \
            --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
            > $log 2>&1
    nums=$(grep -E "^${tag}\b" "$log" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
    echo -e "${alpha}\t${sc}\t${thr}\t${nums}" >> "$OUT"
    echo "  → $nums"
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
