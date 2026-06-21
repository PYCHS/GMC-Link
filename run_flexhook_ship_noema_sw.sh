#!/bin/bash
# Re-validate FlexHook V1 + V2 APPEAR-axis ship recipes on no-EMA + sharedweight caches.
# V1 ship: motion(0.65, 10, +3) + appear(1.0, 3.5, +0.9) → prior 53.696 (mlp + w/EMA)
# V2 ship: motion(0.4, 10, +1.3) + appear(1.0, 3.5, +1.2) → prior 42.799 (mlp + w/EMA)
set -e
cd /home/seanachan/GMC-Link

OUT=results/flexhook_ship_noema_sw_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "arch\tseed\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

# Match output tag to parse: a{α}_scale{sc}_thr{thr}_aa{αa}_sca{sca}_thra{thra}
parse_v1() { grep -E "^a0\.65_scale10\.0_thr3\.0_aa1\.0_sca3\.5_thra0\.9\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }
parse_v2() { grep -E "^a0\.4_scale10\.0_thr1\.3_aa1\.0_sca3\.5_thra1\.2\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

echo "═══ FH V1 ship (motion 0.65/10/+3 + appear 1.0/3.5/+0.9) ═══"
for seed in 0 1 2; do
    suffix="_sharedweight_seed${seed}_noema_rawcos"
    log=/tmp/fh_v1_ship_noema_sw_seed${seed}.log
    echo "▶ V1 seed=${seed}"
    GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
        python run_flexhook_phase5_gmc_sweep.py \
            --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
            --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
            > $log 2>&1
    nums=$(parse_v1 $log)
    echo -e "fh_v1\t${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"
done

echo "═══ FH V2 ship (motion 0.4/10/+1.3 + appear 1.0/3.5/+1.2) ═══"
for seed in 0 1 2; do
    suffix="_sharedweight_seed${seed}_noema_rawcos"
    log=/tmp/fh_v2_ship_noema_sw_seed${seed}.log
    echo "▶ V2 seed=${seed}"
    GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
        python run_flexhook_v2_raw_sweep.py \
            --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
            --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
            > $log 2>&1
    nums=$(parse_v2 $log)
    echo -e "fh_v2\t${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
