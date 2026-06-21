#!/bin/bash
# Multi-seed n=3 train + iKUN HOTA for both CLIP fusion variants on reverted baseline.
# Goal: confirm exp41 late-concat +0.203 single-seed reproduces multi-seed (open
# follow-up flagged in project_exp41_hota_revisit_arch_split).
#
# Variants:
#   early (exp39):  --use-clip-feat --fusion-site input_concat --clip-dim 128
#   late  (exp41):  --use-clip-feat --fusion-site late_concat --clip-dim 64
#                   --text-encoder clip:ViT-B-32:datacomp_xl_s13b_b90k
#                   --lang-passthrough --app-proj-dim 256
#
# iKUN ship: --alpha 1.0 --gmc_scale 0.9 --thr 0.17
#            --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10
#            (NO GMC_RAW_COS — caches built with default sigmoid-EMA scale)
set -e
cd /home/seanachan/GMC-Link

OUT=results/clip_fusion_multiseed_ikun_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results experiments
echo -e "variant\tseed\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing → $OUT"

parse_ikun() { grep -E "^a1\.0_scale0\.9_thr0\.17_aa1\.0_sca0\.3_thra0\.1\b" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

for seed in 0 1 2; do
    # ═══ Early (exp39 input_concat clip-dim 128) ═══
    exp_dir=experiments/exp39_clip128_seed${seed}
    weights=${exp_dir}/weights.pth
    suffix=_exp39_clip128_seed${seed}

    if [ ! -f "$weights" ]; then
        mkdir -p "$exp_dir"
        echo "▶ Train early seed=${seed} → $weights"
        python -m gmc_link.train --split v1 --stage 1 \
            --use-clip-feat --fusion-site input_concat --clip-dim 128 \
            --seed $seed --save-path $weights \
            > ${exp_dir}/train.log 2>&1
    else
        echo "  early seed=${seed} weights exist, skip train"
    fi

    for seq in 0005 0011 0013; do
        cache_path=gmc_link/gmc_scores_v1_${seq}${suffix}_cache.json
        if [ ! -f "$cache_path" ]; then
            echo "  → cache early seed=${seed} seq=${seq}"
            GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                python run_build_gmc_cache.py $seq \
                > ${exp_dir}/cache_${seq}.log 2>&1
        fi
    done

    echo "▶ HOTA early seed=${seed}"
    log=/tmp/clip_mseed_early_seed${seed}.log
    GMC_SUFFIX=$suffix \
        python run_ikun_linear_additive.py \
            --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
            --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
            > $log 2>&1
    nums=$(parse_ikun $log)
    echo -e "early\t${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"

    # ═══ Late (exp41 late_concat cliptext) ═══
    exp_dir=experiments/exp41_late_concat_cliptext_seed${seed}
    weights=${exp_dir}/weights.pth
    suffix=_exp41_lateconcat_seed${seed}

    if [ ! -f "$weights" ]; then
        mkdir -p "$exp_dir"
        echo "▶ Train late seed=${seed} → $weights"
        python -m gmc_link.train --split v1 --stage 1 \
            --use-clip-feat --fusion-site late_concat --clip-dim 64 \
            --text-encoder clip:ViT-B-32:datacomp_xl_s13b_b90k \
            --lang-passthrough --app-proj-dim 256 \
            --seed $seed --save-path $weights \
            > ${exp_dir}/train.log 2>&1
    else
        echo "  late seed=${seed} weights exist, skip train"
    fi

    for seq in 0005 0011 0013; do
        cache_path=gmc_link/gmc_scores_v1_${seq}${suffix}_cache.json
        if [ ! -f "$cache_path" ]; then
            echo "  → cache late seed=${seed} seq=${seq}"
            GMC_WEIGHTS=$weights GMC_SUFFIX=$suffix \
                python run_build_gmc_cache.py $seq \
                > ${exp_dir}/cache_${seq}.log 2>&1
        fi
    done

    echo "▶ HOTA late seed=${seed}"
    log=/tmp/clip_mseed_late_seed${seed}.log
    GMC_SUFFIX=$suffix \
        python run_ikun_linear_additive.py \
            --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
            --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 \
            > $log 2>&1
    nums=$(parse_ikun $log)
    echo -e "late\t${seed}\t${nums}" >> "$OUT"
    echo "  → $nums"
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
