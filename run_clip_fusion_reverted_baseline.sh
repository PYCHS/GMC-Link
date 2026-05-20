#!/bin/bash
# Test CLIP early (exp39 input concat) + late (exp41 aligner-internal concat) fusion
# at current ship recipes on reverted baseline (mlp + w/EMA).
# Phase 1: single-seed cache re-run for reproducibility + 3-arch breakdown.
set -e
cd /home/seanachan/GMC-Link

OUT=results/clip_fusion_reverted_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "fusion\tarch\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

# Ship recipes per arch (locked, mlp+EMA + raw cos GMC cache).
parse_ikun() { grep -E "^a1\.0_scale0\.9_thr0\.17_aa1\.0_sca0\.3_thra0\.1\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }
parse_v1()   { grep -E "^a0\.65_scale10\.0_thr3\.0_aa1\.0_sca3\.5_thra0\.9\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }
parse_v2()   { grep -E "^a0\.4_scale10\.0_thr1\.3_aa1\.0_sca3\.5_thra1\.2\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

run_ikun() {
    local suffix=$1 tag=$2
    local log=/tmp/clip_${tag}_ikun.log
    GMC_SUFFIX=$suffix \
        python run_ikun_linear_additive.py \
            --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
            --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
            > $log 2>&1
    parse_ikun $log
}

run_v1() {
    local suffix=$1 tag=$2
    local log=/tmp/clip_${tag}_fhv1.log
    GMC_SUFFIX=$suffix \
        python run_flexhook_phase5_gmc_sweep.py \
            --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
            --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
            > $log 2>&1
    parse_v1 $log
}

run_v2() {
    local suffix=$1 tag=$2
    local log=/tmp/clip_${tag}_fhv2.log
    GMC_SUFFIX=$suffix \
        python run_flexhook_v2_raw_sweep.py \
            --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
            --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
            > $log 2>&1
    parse_v2 $log
}

echo "═══ exp39 early-concat CLIP128 (input-side) ═══"
echo "▶ iKUN"; nums=$(run_ikun _exp39_clip128 early); echo -e "early\tikun\t${nums}" >> "$OUT"; echo "  → $nums"
echo "▶ FH V1"; nums=$(run_v1 _exp39_clip128 early); echo -e "early\tfh_v1\t${nums}" >> "$OUT"; echo "  → $nums"
echo "▶ FH V2"; nums=$(run_v2 _exp39_clip128 early); echo -e "early\tfh_v2\t${nums}" >> "$OUT"; echo "  → $nums"

echo "═══ exp41 late-concat-cliptext (aligner-internal) ═══"
echo "▶ iKUN"; nums=$(run_ikun _exp41_lateconcat late); echo -e "late\tikun\t${nums}" >> "$OUT"; echo "  → $nums"
echo "▶ FH V1"; nums=$(run_v1 _exp41_lateconcat late); echo -e "late\tfh_v1\t${nums}" >> "$OUT"; echo "  → $nums"
echo "▶ FH V2(no cache)"; echo -e "late\tfh_v2\t-\t-\t-\t-" >> "$OUT"; echo "  → skipped (no exp41 V2 cache)"

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
