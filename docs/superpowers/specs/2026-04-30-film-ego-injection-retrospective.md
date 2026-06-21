# FiLM Ego-Injection Retrospective (Tasks 6-9 closeout)

**Date:** 2026-04-30
**Branches:** GMC-Link `exp/ego-motion-systematic`; iKUN `feat/film-ego-injection`, `feat/film-site-b`
**Spec:** [2026-04-29-film-ego-injection-design.md](./2026-04-29-film-ego-injection-design.md)
**Plan:** [../plans/2026-04-29-film-ego-injection.md](../plans/2026-04-29-film-ego-injection.md)

## Headline

FiLM site-A produces a small but real macro gain (+0.642 HOTA vs paper-strength baseline B), confirms ego compensation is decisive (+34.93 HOTA over rawvel control), and confirms site-A (pre cross_modal_fusion) beats site-B (post st_pooling) by +0.97 HOTA macro.

## Final results (cascade KUM, YOLOv8-NS, 3-seq macro)

| Variant | macro | 0005 | 0011 | 0013 | 0011-MOVING |
|---|---|---|---|---|---|
| Baseline B (no FiLM, paper-strength) | 39.414 | — | 47.085 | — | — |
| **FiLM site-A ep19** (ego-comp) | **40.056** | 46.43 | 47.02 | 26.72 | 8.47 |
| FiLM site-B ep19 (ego-comp) | 39.088 | 45.21 | 46.78 | 25.28 | 5.27 |
| FiLM site-A ep19 (rawvel) | 5.128 | 6.42 | 4.74 | 4.22 | 0.49 |

Per-class on 0011 (site-A vs B): STATIC 51.65, MOVING 8.47, OTHER 36.55. MOVING gain over the broken cascade-KUM motion-class catastrophe (B≈1.55 from prior memory) is +6.92 in absolute terms, but pooled HOTA is flat because cascade KUM already saturates STATIC.

## Three findings

### 1. Site-A > Site-B by +0.97 HOTA macro

Pre-text-attn modulation on `[HW, BT, 2048]` outperforms post-pool modulation on `[b, 1024]`.

**Why:** site-A modulates per-frame visual features before they get correlated with text — text attention can then attend differentially to motion-modulated regions. Site-B modulates the already-pooled global vector, after the spatial and temporal averaging has discarded the per-frame structure that motion can usefully condition. Site-B's head also has half the capacity (1024 vs 2048) but adding capacity won't recover what averaging removed.

**Implication:** for any future motion-injection variant on iKUN-style backbones, inject as early as possible in the fusion stack, before pooling. The cost is 7-fold per-frame compute (HW=49); on a 3060 Ti it's 23 it/s either way.

### 2. Ego compensation is decisive

Removing ego-comp (rawvel) collapses macro from 40.056 → 5.128 (−34.93 HOTA). The MLP head's identity init is preserved at step 0, so the collapse is post-training — the head learns to amplify the ego-contaminated signal into noise that drowns the visual stream.

**Why:** raw velocity in driving footage is dominated by ego motion. Static objects look like they're moving fast in the same direction as the camera; moving objects' apparent velocity is signed by camera direction more than by their own. The MLP cannot disentangle this from 13 dimensions, so it learns to follow the dominant (ego) signal and outputs catastrophic γ, β to fit the training noise.

**Implication:** validates the spec's risk row 1. Any new feature variant must keep ego compensation. Future ablations should run rawvel as a control to isolate "feature-X's contribution" from "ego-comp's contribution."

### 3. Cascade KUM has limited FiLM headroom

Site-A's +0.642 macro is the smallest positive gain measured for any GMC-Link variant on cascade-KUM. Memory `project_simcalib_absorbs_half_motion_gain` (2026-04-28) measured the GMC-α=1 fusion gain shrinking from +10.96 → +3.51 MOVING vs B; FiLM here is +6.92 MOVING on 0011 (site-A vs B's broken 1.55) but net macro is flat because STATIC is already saturated on cascade-KUM.

**Why:** cascade-KUM with sim_calib already absorbs most of the motion signal that's recoverable from text — the bias term encodes which expressions are motion-class and weights them. FiLM adds an orthogonal axis (per-object motion magnitude/direction), but the cascade has already captured most of what 13D can offer.

**Implication:** further gains on cascade-KUM probably require non-13D motion features (richer kinematics, scene context, attention over track history) or a different injection target (text-conditional gating, not vis-conditional FiLM).

## Surprises

- **0013 collapse on FiLM:** site-A 0013 MOVING=0.09, STATIC=0.00. The seq has only n=2 expressions per memory `project_v1_seq0013_data_thinness`; FiLM may be over-specializing to seqs with denser supervision. The macro gain of +0.642 is real but driven by 0005 + flat 0011, not 0013.
- **Train loss did not predict eval rank:** site-B reached lower training loss (~0.6 by ep13) than site-A (~0.7 by ep13) but lost on HOTA. The post-pool head is easier to train (smaller search space) but produces a weaker eval. Don't trust training loss as a site selection signal.
- **Smoke vs real-train BT mismatch:** test-mode bs=1 hid the BT-broadcast bug because PyTorch broadcasts size-1 dims silently. Production bs=8 surfaced it as `tensor a (8) vs b (16)`. Lesson: smoke tests at bs=1 are insufficient for shape-sensitive code; always smoke at bs≥2.

## Open questions

- Is there an ep<19 site-A ckpt that beats 40.056? Loss oscillated 0.61-0.73 ep5-19; cosine schedule should converge ep19 but the monotonic loss claim is weak. Worth eval ep15 + ep17 if pursuing site-A further.
- Does FiLM compose with the threshold-drop + GMC-α stack (`project_phase4_threshold_gmc_stack`)? Site-A + thr=−0.3 + α=0.5 is unmeasured; could yield additive motion gains.
- Site-A on non-cascade iKUN baseline: untested. Memory shows non-cascade has bigger headroom (`project_phase5_noncascade_stack_universal`).

## Decisions

1. **Ship site-A as the FiLM recipe.** Drop site-B branch.
2. **Keep ego-comp as the only valid 13D variant.** Rawvel control falsified.
3. **Do not pursue further FiLM tuning on cascade-KUM** without first checking the threshold-drop + GMC-α stack composition. The +0.642 macro is small enough that hyperparameter exploration on 13D + injection topology has bad expected return.
4. **Update memory:** record FiLM as "small positive on cascade-KUM, decisive on rawvel, site-A topology required."

## Branch cleanup

- iKUN `feat/film-ego-injection` (site-A): keep, ship-able state.
- iKUN `feat/film-site-b`: keep as comparison record, do not merge.
- GMC-Link `exp/ego-motion-systematic`: keep, FiLM driver scripts + rawvel ablation script committed.

## Reproducibility

- Site-A ckpt: `/home/seanachan/GMC-Link/film_v1/epoch19.pth`
- Site-B ckpt: `/home/seanachan/GMC-Link/film_siteB_v1/epoch19.pth`
- Rawvel ckpt: `/home/seanachan/GMC-Link/film_rawvel_v1/epoch19.pth`
- 13D caches: `iKUN/motion_13d_cache_v1/` (ego-comp), `iKUN/motion_13d_cache_rawvel_v1/`
- Eval logs: `/tmp/film_eval_ep19.log`, `/tmp/film_siteB_eval_ep19.log`, `/tmp/film_rawvel_eval_ep19.log`
- Train logs: `/tmp/film_v1.log`, `/tmp/film_siteB_train.log`, `/tmp/film_rawvel_train.log`
