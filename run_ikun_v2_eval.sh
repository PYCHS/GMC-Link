#!/bin/bash
# iKUN-V2 fusion eval (cross-split grid completion). Two-baseline protocol:
#   B1 = iKUN-V2 cascade-only (no GMC, alpha=0)
#   ship = iKUN-V2 cascade + GMC sw-aligner linear-additive (V1 ship recipe)
# V2 expressions over NeuralSORT 3-seq, FH gt_template_gen GT, sim-calib OFF
# (V1 text-feat → bias=0 for V2 paraphrases). Single seed (seed0) first pass.
set -e
cd /home/seanachan/GMC-Link

export IKUN_DATA_ROOT=/home/seanachan/data/Dataset/refer-kitti-v2
export IKUN_GT_TEMPLATE=/home/seanachan/FlexHook/datasets/refer-kitti-v2/gt_template_gen
export IKUN_CASCADE_JSON=/home/seanachan/GMC-Link/iKUN/ikun_results_v2_cascade_full.json
export IKUN_OUT_ROOT=/home/seanachan/GMC-Link/hota_eval_ikun_v2
export GMC_CACHE_VER=v2
export GMC_SUFFIX=_sharedweight_seed0_rawcos
export GMC_RAW_COS=1

echo "═══ B1: iKUN-V2 cascade-only (alpha=0) ═══"
python run_ikun_linear_additive.py --alpha 0 --gmc_scale 0 --thr 0 2>&1 | tail -6

echo ""
echo "═══ ship: iKUN-V2 cascade + GMC (V1 recipe α=1.0 sc=0.9 thr=0.17 + appear 1.0/0.30/0.10) ═══"
python run_ikun_linear_additive.py \
    --alpha 1.0 --gmc_scale 0.9 --thr 0.17 \
    --alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10 2>&1 | tail -6
