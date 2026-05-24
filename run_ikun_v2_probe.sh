#!/bin/bash
# iKUN-V2 recipe probe (A). Baseline (NSconv GT) = 31.434. V1 recipe NEG (-2.78).
# Isolate bias-transfer (thr) vs GMC-signal-fails. Probe thr=0 + gentler scales.
set -e
cd /home/seanachan/GMC-Link
export IKUN_DATA_ROOT=/home/seanachan/data/Dataset/refer-kitti-v2
export IKUN_GT_TEMPLATE=/home/seanachan/GMC-Link/v2_gt_template_nsconv
export IKUN_CASCADE_JSON=/home/seanachan/GMC-Link/iKUN/ikun_results_v2_cascade_full.json
export IKUN_OUT_ROOT=/home/seanachan/GMC-Link/hota_eval_ikun_v2_probe
export GMC_CACHE_VER=v2
export GMC_SUFFIX=_sharedweight_seed0_rawcos
export GMC_RAW_COS=1

probe() {  # name am scm thrm aa sca thra
  echo "--- $1: motion($2,$3,$4) appear($5,$6,$7) ---"
  python run_ikun_linear_additive.py \
    --alpha $2 --gmc_scale $3 --thr $4 \
    --alpha_appear $5 --gmc_scale_appear $6 --thr_appear $7 2>&1 | grep -aE "^a${2}_scale"
}

# P1: V1 scales, NO bias (isolate thr-transfer)
probe P1_nobias      1.0 0.9 0.0  1.0 0.30 0.0
# P2: gentler scales, no bias
probe P2_gentle      1.0 0.5 0.0  1.0 0.15 0.0
# P3: lower motion alpha, no bias
probe P3_loweralpha  0.5 0.9 0.0  1.0 0.30 0.0
# P4: motion-only (no appear axis), no bias
probe P4_motiononly  1.0 0.9 0.0  0.0 0.0  0.0
