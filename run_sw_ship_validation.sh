#!/bin/bash
# 2026-05-21 sw + w/EMA full ship validation across 3 archs × 3 seeds.
# Prior sw ship test (project_ikun_ship_noema_sw / project_flexhook_ship_noema_sw)
# bundled sw + no-EMA → V1 regressed −0.190. EMA reverted via commit 8225022.
# This re-tests sw aligner under CURRENT defaults (w/EMA, sigmoid).
#
# Ship recipes (locked, per project_*_multiseed):
#   iKUN: α=1.0 sc=0.9 thr=+0.17 + α_a=1.0 sc_a=0.30 thr_a=+0.10
#   FH V1: α=0.65 sc=10 thr=+3 + α_a=1.0 sc_a=3.5 thr_a=+0.9
#   FH V2: α=0.4 sc=10 thr=+1.3 + α_a=1.0 sc_a=3.5 thr_a=+1.2
#
# Ship-swap decision rule: sw POS or NEU on all 3 archs (no arch regresses
# beyond ±0.05) → swap to sw. Otherwise stay mlp.
set -e
cd /home/seanachan/GMC-Link

OUT=results/sw_ship_validation_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "arch\tseed\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

parse_ikun() { grep -E "^a1\.0_scale0\.9_thr0\.17_aa1\.0_sca0\.3_thra0\.1\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }
parse_v1()   { grep -E "^a0\.65_scale10\.0_thr3\.0_aa1\.0_sca3\.5_thra0\.9\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }
parse_v2()   { grep -E "^a0\.4_scale10\.0_thr1\.3_aa1\.0_sca3\.5_thra1\.2\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

V1_SEQS="0005 0011 0013"
V2_SEQS="0005 0011 0013 0019"

for seed in 0 1 2; do
    weights=gmc_link_weights_v1train_sharedweight_seed${seed}.pth
    suffix=_sharedweight_seed${seed}
    log_dir=/tmp/sw_ship_seed${seed}
    mkdir -p $log_dir

    # ─── iKUN caches (3 V1 seqs) ───
    for seq in $V1_SEQS; do
        cache=gmc_link/gmc_scores_v1_${seq}${suffix}_cache.json
        if [ ! -f "$cache" ]; then
            echo "  → cache iKUN seed=${seed} seq=${seq}"
            GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                python run_build_gmc_cache.py $seq \
                > ${log_dir}/cache_ikun_${seq}.log 2>&1
        fi
    done
    echo "▶ HOTA iKUN seed=${seed}"
    log=${log_dir}/hota_ikun.log
    GMC_SUFFIX=$suffix \
        python run_ikun_linear_additive.py \
            --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
            --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
            > $log 2>&1
    nums=$(parse_ikun $log)
    echo -e "ikun\t${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"

    # ─── FH V1 caches (3 V1 seqs) ───
    for seq in $V1_SEQS; do
        cache=gmc_link/gmc_scores_flexhook_v1_${seq}${suffix}_cache.json
        if [ ! -f "$cache" ]; then
            echo "  → cache FH V1 seed=${seed} seq=${seq}"
            GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                python run_build_gmc_cache_flexhook.py $seq \
                > ${log_dir}/cache_v1_${seq}.log 2>&1
        fi
    done
    echo "▶ HOTA FH V1 seed=${seed}"
    log=${log_dir}/hota_v1.log
    GMC_SUFFIX=$suffix \
        python run_flexhook_phase5_gmc_sweep.py \
            --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
            --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
            > $log 2>&1
    nums=$(parse_v1 $log)
    echo -e "fh_v1\t${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"

    # ─── FH V2 caches (4 V2 seqs) ───
    for seq in $V2_SEQS; do
        cache=gmc_link/gmc_scores_flexhook_v2_raw_${seq}${suffix}_cache.json
        if [ ! -f "$cache" ]; then
            echo "  → cache FH V2 seed=${seed} seq=${seq}"
            GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                python run_build_gmc_cache_flexhook_v2_raw.py $seq \
                > ${log_dir}/cache_v2_${seq}.log 2>&1
        fi
    done
    echo "▶ HOTA FH V2 seed=${seed}"
    log=${log_dir}/hota_v2.log
    GMC_SUFFIX=$suffix \
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
