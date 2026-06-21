#!/bin/bash
# Multi-seed n=3 CLIP fusion HOTA on FlexHook V1 + V2 (reuse seed 0/1/2 weights
# from run_clip_fusion_multiseed_ikun.sh). Variants: exp39 input-concat 128D,
# exp41 late-concat-cliptext.
#
# Ship recipes:
#   FH V1: motion(0.65, 10, +3) + appear(1.0, 3.5, +0.9)
#   FH V2: motion(0.4, 10, +1.3) + appear(1.0, 3.5, +1.2)
set -e
cd /home/seanachan/GMC-Link

OUT=results/clip_fusion_multiseed_flexhook_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "variant\tarch\tseed\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

parse_v1() { grep -E "^a0\.65_scale10\.0_thr3\.0_aa1\.0_sca3\.5_thra0\.9\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }
parse_v2() { grep -E "^a0\.4_scale10\.0_thr1\.3_aa1\.0_sca3\.5_thra1\.2\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

V1_SEQS="0005 0011 0013"
V2_SEQS="0005 0011 0013 0019"

for seed in 0 1 2; do
    for variant in early late; do
        if [ "$variant" = "early" ]; then
            weights=experiments/exp39_clip128_seed${seed}/weights.pth
            suffix=_exp39_clip128_seed${seed}
        else
            weights=experiments/exp41_late_concat_cliptext_seed${seed}/weights.pth
            suffix=_exp41_lateconcat_seed${seed}
        fi
        log_dir=$(dirname $weights)

        # ═══ FH V1 cache + HOTA ═══
        for seq in $V1_SEQS; do
            cache=gmc_link/gmc_scores_flexhook_v1_${seq}${suffix}_cache.json
            if [ ! -f "$cache" ]; then
                echo "  → cache ${variant} fh_v1 seed=${seed} seq=${seq}"
                GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                    python run_build_gmc_cache_flexhook.py $seq \
                    > ${log_dir}/cache_v1_${seq}.log 2>&1
            fi
        done
        echo "▶ HOTA ${variant} fh_v1 seed=${seed}"
        log=/tmp/clip_fh_v1_${variant}_seed${seed}.log
        GMC_SUFFIX=$suffix \
            python run_flexhook_phase5_gmc_sweep.py \
                --alpha 0.65 --gmc_scale 10.0 --thr 3.0 \
                --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 0.9 \
                > $log 2>&1
        nums=$(parse_v1 $log)
        echo -e "${variant}\tfh_v1\t${seed}\t${nums}" >> "$OUT"
        echo "  → $nums"

        # ═══ FH V2 cache + HOTA ═══
        for seq in $V2_SEQS; do
            cache=gmc_link/gmc_scores_flexhook_v2_raw_${seq}${suffix}_cache.json
            if [ ! -f "$cache" ]; then
                echo "  → cache ${variant} fh_v2 seed=${seed} seq=${seq}"
                GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                    python run_build_gmc_cache_flexhook_v2_raw.py $seq \
                    > ${log_dir}/cache_v2_${seq}.log 2>&1
            fi
        done
        echo "▶ HOTA ${variant} fh_v2 seed=${seed}"
        log=/tmp/clip_fh_v2_${variant}_seed${seed}.log
        GMC_SUFFIX=$suffix \
            python run_flexhook_v2_raw_sweep.py \
                --alpha 0.4 --gmc_scale 10.0 --thr 1.3 \
                --alpha_appear 1.0 --gmc_scale_appear 3.5 --thr_appear 1.2 \
                > $log 2>&1
        nums=$(parse_v2 $log)
        echo -e "${variant}\tfh_v2\t${seed}\t${nums}" >> "$OUT"
        echo "  → $nums"
    done
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
