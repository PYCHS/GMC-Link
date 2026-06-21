#!/bin/bash
# Phase 1 gate: re-eval exp41 seed 0 at current ship recipe (sw+norec+no-EMA convention).
# Build raw_cos caches with exp41 weights, run ship HOTA per arch, check per-class breakdown
# vs current ship (44.634/53.526/42.807) to see if STATIC regression pattern persists.
#
# If STATIC drops vs sw → Phase 2 per-class routing.
# If STATIC matches → close direction.
set -e
cd /home/seanachan/GMC-Link

OUT=results/phase1_exp41_gate_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "arch\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"

WEIGHTS=experiments/exp41_late_concat_cliptext_seed0/weights.pth
SUFFIX=_exp41_lateconcat_seed0_rawcos

export GMC_WEIGHTS=$WEIGHTS
export GMC_SUFFIX=$SUFFIX
export GMC_RAW_COS=1

V1_SEQS="0005 0011 0013"
V2_SEQS="0005 0011 0013 0019"

# ─── iKUN caches + HOTA ───
echo "═══ iKUN exp41 raw_cos ═══"
for seq in $V1_SEQS; do
    cache=gmc_link/gmc_scores_v1_${seq}${SUFFIX}_cache.json
    if [ ! -f "$cache" ]; then
        echo "  → cache iKUN seq=${seq}"
        python run_build_gmc_cache.py $seq > /tmp/phase1_cache_ikun_${seq}.log 2>&1
    fi
done
echo "▶ HOTA iKUN"
log=/tmp/phase1_ikun.log
python run_ikun_linear_additive.py \
    --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
    --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
    > $log 2>&1
nums=$(grep -E "^a1\.0_scale0\.9_thr0\.17_aa1\.0_sca0\.3_thra0\.1\b" $log | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
echo -e "ikun\t${nums}" >> "$OUT"
echo "  → $nums"

# ─── FH V1 caches + HOTA ───
echo "═══ FH V1 exp41 raw_cos ═══"
for seq in $V1_SEQS; do
    cache=gmc_link/gmc_scores_flexhook_v1_${seq}${SUFFIX}_cache.json
    if [ ! -f "$cache" ]; then
        echo "  → cache FH V1 seq=${seq}"
        python run_build_gmc_cache_flexhook.py $seq > /tmp/phase1_cache_fhv1_${seq}.log 2>&1
    fi
done
echo "▶ HOTA FH V1"
log=/tmp/phase1_fhv1.log
python run_flexhook_phase5_gmc_sweep.py \
    --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
    --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
    > $log 2>&1
nums=$(grep -E "^a0\.65_scale10\.0_thr3\.0_aa1\.0_sca3\.5_thra0\.9\b" $log | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
echo -e "fh_v1\t${nums}" >> "$OUT"
echo "  → $nums"

# ─── FH V2 caches + HOTA ───
echo "═══ FH V2 exp41 raw_cos ═══"
for seq in $V2_SEQS; do
    cache=gmc_link/gmc_scores_flexhook_v2_raw_${seq}${SUFFIX}_cache.json
    if [ ! -f "$cache" ]; then
        echo "  → cache FH V2 seq=${seq}"
        python run_build_gmc_cache_flexhook_v2_raw.py $seq > /tmp/phase1_cache_fhv2_${seq}.log 2>&1
    fi
done
echo "▶ HOTA FH V2"
log=/tmp/phase1_fhv2.log
python run_flexhook_v2_raw_sweep.py \
    --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
    --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
    > $log 2>&1
nums=$(grep -E "^a0\.4_scale10\.0_thr1\.3_aa1\.0_sca3\.5_thra1\.2\b" $log | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
echo -e "fh_v2\t${nums}" >> "$OUT"
echo "  → $nums"

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
