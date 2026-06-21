#!/bin/bash
# Re-validate iKUN APPEAR-axis ship recipe on no-EMA + sharedweight aligner caches.
# Ship recipe (locked):
#   motion:  alpha_m=1.0, sc_m=0.9,  thr_m=+0.17
#   appear:  alpha_a=1.0, sc_a=0.30, thr_a=+0.10
# Prior 44.602 result was on mlp + w/EMA.  Re-test under new defaults (sharedweight + no-EMA).
set -e
cd /home/seanachan/GMC-Link

OUT=results/ikun_appearship_noema_sw_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "seed\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

parse_hota() { grep -E "^a1\.0_scale0\.9_thr0\.17_aa1\.0_sca0\.3_thra0\.1\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

for seed in 0 1 2; do
    suffix="_sharedweight_seed${seed}_noema_rawcos"
    log=/tmp/ikun_appearship_noema_sw_seed${seed}.log
    echo "▶ seed=${seed} (suffix=${suffix})"
    GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
        python run_ikun_linear_additive.py \
            --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
            --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
            > $log 2>&1
    nums=$(parse_hota $log)
    echo -e "${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
