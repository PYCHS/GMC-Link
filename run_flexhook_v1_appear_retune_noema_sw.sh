#!/bin/bash
# V1 APPEAR-axis retune on sw+no-EMA. Motion locked at (0.65, 10, +3).
# APPEAR axis is 77% of V1 frames; if pool regressed -0.190, APPEAR axis
# is likely lever (motion-axis sweep exhausted).
set -e
cd /home/seanachan/GMC-Link

OUT=results/flexhook_v1_appear_retune_noema_sw_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "alpha_a\tsc_a\tthr_a\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

suffix="_sharedweight_seed0_noema_rawcos"

# Sweep appear axis. Motion locked (0.65, 10, +3). Current ship: (1.0, 3.5, +0.9).
declare -a CONFIGS=(
    "1.0 3.5 0.9"   # control (current ship)
    "1.0 2.5 0.9"   # lower sc_a
    "1.0 3.0 0.9"
    "1.0 4.0 0.9"   # higher sc_a
    "1.0 4.5 0.9"
    "1.0 3.5 0.7"   # lower thr_a
    "1.0 3.5 1.1"   # higher thr_a
    "1.0 4.0 1.1"   # higher both
    "1.0 2.5 0.7"   # both lower
)

for cfg in "${CONFIGS[@]}"; do
    read -r aa sa ta <<< "$cfg"
    tag="a0.65_scale10.0_thr3.0_aa${aa}_sca${sa}_thra${ta}"
    log=/tmp/fh_v1_appear_retune_${aa}_${sa}_${ta}.log
    echo "▶ αa=${aa} sca=${sa} thra=${ta}"
    GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
        python run_flexhook_phase5_gmc_sweep.py \
            --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
            --alpha_appear $aa --gmc_scale_appear $sa --thr_appear $ta \
            > $log 2>&1
    nums=$(grep -E "^${tag}\b" "$log" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
    echo -e "${aa}\t${sa}\t${ta}\t${nums}" >> "$OUT"
    echo "  → $nums"
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
