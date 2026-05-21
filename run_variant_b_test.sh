#!/bin/bash
# Variant B HOTA test: derived sc (from std-matching) + α=1.0 + thr=0.
# Single seed. If clear NEG, falsifies direction; if POS, expand to n=3.
set -e
cd /home/seanachan/GMC-Link
OUT=results/variant_b_test_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "arch\tsc_m\tsc_a\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"

GMC_SUFFIX=_sharedweight_seed0_rawcos
export GMC_RAW_COS=1
export GMC_SUFFIX

# iKUN: sc_m=1.06, sc_a=2.15
log=/tmp/variant_b_ikun.log
python run_ikun_linear_additive.py \
    --alpha 1.0 --gmc_scale 1.06 --thr 0.0 \
    --alpha_appear 1.0 --gmc_scale_appear 2.15 --thr_appear 0.0 \
    > $log 2>&1
nums=$(grep -E "^a1\.0_scale1\.06_thr0\.0_aa1\.0_sca2\.15_thra0\.0\b" $log | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
echo -e "ikun\t1.06\t2.15\t${nums}" >> "$OUT"
echo "iKUN  sc(1.06, 2.15) → $nums"

# FH V1: sc_m=29.97, sc_a=39.52
log=/tmp/variant_b_fhv1.log
python run_flexhook_phase5_gmc_sweep.py \
    --alpha 1.0 --gmc_scale 29.97 --thr 0.0 \
    --alpha_appear 1.0 --gmc_scale_appear 39.52 --thr_appear 0.0 \
    > $log 2>&1
nums=$(grep -E "^a1\.0_scale29\.97_thr0\.0_aa1\.0_sca39\.52_thra0\.0\b" $log | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
echo -e "fh_v1\t29.97\t39.52\t${nums}" >> "$OUT"
echo "FH V1 sc(29.97, 39.52) → $nums"

# FH V2: sc_m=38.65, sc_a=39.44
log=/tmp/variant_b_fhv2.log
python run_flexhook_v2_raw_sweep.py \
    --alpha 1.0 --gmc_scale 38.65 --thr 0.0 \
    --alpha_appear 1.0 --gmc_scale_appear 39.44 --thr_appear 0.0 \
    > $log 2>&1
nums=$(grep -E "^a1\.0_scale38\.65_thr0\.0_aa1\.0_sca39\.44_thra0\.0\b" $log | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}')
echo -e "fh_v2\t38.65\t39.44\t${nums}" >> "$OUT"
echo "FH V2 sc(38.65, 39.44) → $nums"

echo ""; cat "$OUT"
