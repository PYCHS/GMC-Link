#!/bin/bash
# Two-baseline protocol AFTER dropping all EMA (MotionBuffer + ScoreBuffer + cosine_buffer).
# Compare against committed shared_weight numbers (which still had MotionBuffer EMA active).
# Same recipe: --alpha 1 --gmc_scale 1 --thr 0, raw cos.
# Suffix uses _noema_ to preserve prior caches for side-by-side compare.
set -e
cd /home/seanachan/GMC-Link

OUT=results/noema_two_baseline_$(date +%Y%m%d_%H%M%S).tsv
mkdir -p results
echo -e "arch\taligner\tseed\trecipe\tpooled\tAPPEAR\tMOVING\tSTATIC" > "$OUT"
echo "Writing results → $OUT"

arch_cache() { case $1 in ikun) echo "run_build_gmc_cache.py";; fh_v1) echo "run_build_gmc_cache_flexhook.py";; fh_v2) echo "run_build_gmc_cache_flexhook_v2_raw.py";; esac; }
arch_driver() { case $1 in ikun) echo "run_ikun_linear_additive.py";; fh_v1) echo "run_flexhook_phase5_gmc_sweep.py";; fh_v2) echo "run_flexhook_v2_raw_sweep.py";; esac; }
arch_seqs() { case $1 in ikun) echo "0005 0011 0013";; fh_v1) echo "0005 0011 0013";; fh_v2) echo "0005 0011 0013 0019";; esac; }
aligner_weight() {
    local aligner=$1 seed=$2
    case $aligner in
        mlp) echo "gmc_link_weights_v1train_seed${seed}.pth";;
        sharedweight) echo "gmc_link_weights_v1train_sharedweight_seed${seed}.pth";;
    esac
}
parse_hota() { grep -E "^a[01]\.[0-9]+_scale" "$1" | tail -1 | awk '{print $(NF-3)"\t"$(NF-2)"\t"$(NF-1)"\t"$NF}'; }

# B1: deterministic (no GMC, EMA changes don't affect)
echo "═══ B1 baselines (unchanged by EMA drop) ═══"
for arch in ikun fh_v1 fh_v2; do
    driver=$(arch_driver $arch)
    echo "▶ $arch B1"
    log=/tmp/b1_${arch}_noema.log
    python $driver --alpha 0 --gmc_scale 0 --thr 0 > $log 2>&1
    nums=$(parse_hota $log)
    echo -e "$arch\t-\t-\t{model} Baseline\t$nums" >> "$OUT"
    echo "  → $nums"
done

# B2: rebuild caches with noema suffix
echo "═══ B2 cells (no-EMA aligner output) ═══"
for arch in ikun fh_v1 fh_v2; do
    builder=$(arch_cache $arch)
    driver=$(arch_driver $arch)
    seqs=$(arch_seqs $arch)
    for aligner in mlp sharedweight; do
        for seed in 0 1 2; do
            weight=$(aligner_weight $aligner $seed)
            suffix="_${aligner}_seed${seed}_noema_rawcos"
            echo "▶ $arch + $aligner seed=$seed (no-EMA)"
            GMC_WEIGHTS=$weight GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
                python $builder $seqs > /tmp/cache_${arch}_${aligner}_${seed}_noema.log 2>&1
            log=/tmp/b2_${arch}_${aligner}_${seed}_noema.log
            GMC_SUFFIX=$suffix GMC_RAW_COS=1 \
                python $driver --alpha 1 --gmc_scale 1 --thr 0 > $log 2>&1
            nums=$(parse_hota $log)
            echo -e "$arch\t$aligner\t$seed\t{model} + GMC Baseline (no EMA)\t$nums" >> "$OUT"
            echo "  → $nums"
        done
    done
done

echo ""
echo "=== Done. Results in $OUT ==="
cat "$OUT"
