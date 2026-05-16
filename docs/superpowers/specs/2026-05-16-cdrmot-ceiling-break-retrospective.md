# CDRMOT-Inspired Ceiling-Break Retrospective — 2026-05-16

## TL;DR

Two transferable mechanisms from CDRMOT (Liang et al., arXiv 2503.11496) tested as
the final ceiling-break attempts on the GMC-Link iKUN ship recipe (paper-canonical
3-seq pooled HOTA on Refer-KITTI V1, ship multi-seed n=3 mean **44.608 ± 0.024**,
already beating paper YOLOv8-NS 44.564 at p=0.044).

- **Lever A — Structural Consensus Constraint** (aux loss preserving pairwise distance
  between motion and language manifolds): robust NEG across λ_struct ∈ {0.1, 0.5}.
  Best λ=0.1 pooled3=44.434 (Δship=−0.152). Manifold-collapse failure mode confirmed.
- **Lever B — What/Where Dual Cosine** (spaCy POS-decompose RMOT expressions into
  what/where tokens, run cascade-attention twice, linear-fuse logits): NEG far below
  kill gate across w_what ∈ {0.3, 0.5, 0.7}. Best w_what=0.5 pooled3=40.918
  (Δship=−3.67, ~150σ below ship multi-seed). Signal destruction by stub-text inputs.

**No new ship.** iKUN appear-ship multi-seed 44.608 remains the live recipe and
the honest ceiling claim. Decision tree terminal: A1 NEG ∧ B1 NEG → writeup.

This closes the 19th lever at the 44.608 ceiling.

## Context

After 18 levers exhausted at the GMC-Link iKUN ship ceiling (representation,
architecture, fusion site, supervision expansion, language encoder, curriculum,
ego source, per-class specialist, depth-aug, world-XY, CLIP-visual concat,
CLIP-logit decision fusion, Case 2 fusion-transformer 1a/1b/1c/1d, ARM B raw
cosine, threshold and Phase 5 stacks), CDRMOT was reviewed for transferable
mechanisms. Two were untested in our pipeline:

1. **Structural Consensus Constraint** — geometry regularizer aux loss. Different
   lever class from InfoNCE shape (Exp 34 HN-InfoNCE β-grid proved 0.779 AUC ceiling
   is representation-bound at loss-shape level; geometry-matching aux was a
   distinct untested class).
2. **What/Where Dual Cosine** — score-side text decomposition. Genuinely new vs
   ship `is_motion_flag` (per-expression class flag, not per-token POS split).

DRMOT (arXiv 2602.04692) was also reviewed but contributed nothing transferable —
pseudo-RGB depth fusion and Hungarian depth cost both live at tracker/backbone
level, blocked by NeuralSORT upstream wall.

## Lever A: Structural Consensus Constraint

### Hypothesis

Adding `L_struct = MSE(D_norm(z_m), D_norm(z_l))` to InfoNCE training (where
`D_norm = cdist / cdist.mean()` is scale-normalized pairwise euclidean) forces
the motion-manifold relative geometry to match the language-manifold relative
geometry across each batch. Because Exp 34 already proved the ceiling is not
loss-shape-bound, only an aux objective of a *different class* could break it.

### Implementation

Files modified (kept in tree, disabled by default):

- `gmc_link/losses.py:72-130` — `StructuralConsensusLoss(mode={dist,dist_angle})`.
  Pairwise-distance MSE with optional triplet-angle term.
- `gmc_link/train.py:96-117, 367-372, 519-528, 620+` — wire aux loss via
  `model.encode()` (no `forward()` API break), CLI flags
  `--lam-struct --lam-angle --struct-mode`.
- `gmc_link/manager.py:86-87` — fix latent ckpt-loading bug
  (`int(checkpoint.get("clip_feat_dim", 512))` failed on new ckpts serializing
  `clip_feat_dim=None`). Switched to `int(... or 512)`.

### Training and Eval

- Phase A1 single-seed runs (V1 split, 100ep, seed=1, dist-only).
- λ_struct=0.5: pooled3=44.376 (Δship=−0.210), MOVING −1.99, STATIC −1.99.
- λ_struct=0.1: pooled3=44.434 (Δship=−0.152), STATIC ≈ baseline, MOVING −2.75.

### Verdict

**Robust kill across regularizer strength.** Monotone-NEG slope λ=0 → λ=0.1 →
λ=0.5: 44.586 → 44.434 → 44.376. Asymmetric failure: light λ preferentially
distorts moving-cluster geometry; heavy λ collapses both classes uniformly.

**Mechanism — manifold collapse.** InfoNCE creates discriminative cluster
structure; structural consensus enforces global metric-matching. If language
manifold is naturally tighter or more spread than the discriminative motion
structure, the aux loss compresses inter-class margins in the motion embedding
to match language geometry. Final InfoNCE loss 4.25 is higher than typical
stage1 final (~3.8), consistent with aux objective competing with main objective.

CDRMOT structural-loss direction **CLOSED** on iKUN ship recipe.

## Lever B: What/Where Dual Cosine

### Hypothesis

RMOT expressions mix appearance ("red car") and motion ("turning left"). A
single CLIP-text embedding entangles both. Decomposing via spaCy POS, encoding
each branch separately through the existing cascade-attention model, and
linear-fusing the logits should yield better discrimination than the single
entangled cascade pass.

### Implementation

- `gmc_link/text_what_where.py` — spaCy `en_core_web_sm` parser with keyword
  overrides. Tokens classified as:
  - **what**: NOUN/PROPN/ADJ + color keywords (red, white, light, dark, …)
  - **where**: VERB/ADV + spatial keywords (left, behind, near, …) + 26-token
    motion lexicon (moving, turning, walking, …) + prepositions (ADP)
  - Drop: DET, PUNCT, AUX, SCONJ, CCONJ
  - Fallback: if either branch empty, use full expression
- Collected 318 unique V1 expressions across both raw file stems (refer-kitti
  `expression/{seq}/*.json`) and iKUN-converted form (`text_feat_bboxNum_v1.json`).
  305/318 (96%) decomposed into both branches; 13 fell back to full text.
  Output: `iKUN/expr_what_where_v1.json`.

Sample parses:
- `right-moving-cars` → what="car", where="right moving"
- `vehicles-in-the-same-direction-of-ours` → what="car", where="in same direction of ours"
- `walking-males` → what="men", where="walking"

- `run_cascade_v1test_with_text_override.py` — generic cascade-attention driver
  reading `{raw → field}` mapping. Patches `sys.argv` before iKUN import to
  remain a standalone CLI. Output schema matches existing
  `iKUN/ikun_results_v1_cascade_full.json`.
- `run_whatwhere_b1_eval.py` — loads both cache JSONs, fuses
  `cs_fused = w_what·cs_what + w_where·cs_where`, applies LOCKED iKUN ship recipe
  (motion α=1.0 sc=0.9 thr=+0.17, APPEAR α_a=1.0 sc_a=0.30 thr_a=+0.10 + simcalib),
  pipes through `gen_predict` + `run_te`.

### Cascade Inference

Two passes over the iKUN cascade attention model (`iKUN_cascade_attention.pth`)
on V1 test (0005/0011/0013), 66456 batches each, ~46min @ 23 it/s per pass.
Substitutions per pass: 63566 / 2890 identical.

### Results

3-seq pooled HOTA on `gt_template_old`:

| w_what | w_where | pooled | APPEAR | MOVING | STATIC | Δ ship |
|--------|---------|--------|--------|--------|--------|--------|
| 0.30   | 0.70    | 40.308 | 40.848 | 31.501 | 44.859 | −4.28  |
| 0.50   | 0.50    | **40.918** | 42.065 | 31.046 | 44.230 | **−3.67** |
| 0.70   | 0.30    | 40.073 | 41.540 | 30.665 | 42.711 | −4.51  |

Ship seed-1 reference: 44.586. Multi-seed (n=3): 44.608 ± 0.024.

### Verdict

Best cell ~150σ below ship; far below kill gate +0.05. **KILLED.**

**Mechanism — cross-modal signal destruction.** Three failures stacked:

1. **CLIP-text encoder is sentence-trained, not phrase-trained.** Substituting
   `right-moving-cars` → `car` (what) or `right moving` (where) feeds the
   cascade text-attention degenerate single-noun or stub-phrase inputs.
   Embeddings for these stubs do not match the visual-attention prior the
   cascade was trained against on full RMOT phrases.
2. **Linear-additive fusion cannot recover the lost discrimination.** Each
   branch's logits live on shifted scales (single-noun "car" matches every
   car-bbox frame; "moving" matches every motion frame). Fused score is
   dominated by whichever branch saturates — typically the appearance-only
   branch — washing out MOVING discrimination. STATIC ≈ baseline at w_what=0.3
   is consistent: "parked car" → `what=car, where=parked` and the where-branch
   dominating recovers the original signal for static cases.
3. **POS-decomposition is not paraphrase-augmentation.** CDRMOT used
   parser-driven decomposition as input to a *retrained* dual-tower model with
   separate what/where text encoders. Their +6.0% HOTA claim comes from
   architectural specialization, not from running stock CLIP-cascade twice
   with substituted inputs. Our score-side adaptation is fundamentally
   unsupported by the stock cascade.

CDRMOT what/where direction **CLOSED** on iKUN ship recipe.

## Decision Tree Terminal

Plan `valiant-sleeping-mochi.md` terminal trigger:
> "Commit to writeup at 44.608 immediately if (A1 NEG ∧ B1 NEG)."

Both NEG. **Trigger met.**

## Ceiling-Break Campaign Summary (19 Levers)

Final state of the 44.608 ceiling investigation. All levers tested at the
iKUN ship recipe (cascade + simcalib + motion-axis GMC + APPEAR-axis GMC) or
its aligner-stage upstream:

| # | Lever | Class | Verdict |
|---|-------|-------|---------|
| 1 | Exp 36A 25D MLP (scale-diff, temporal-deriv) | features | NEG |
| 2 | Exp 36B transformer 13D | architecture | NEG |
| 3 | Exp 36C V1+V2 joint training | supervision | NEG (micro) |
| 4 | Exp 36D BGE-base 768D language | text encoder | NEG |
| 5 | Exp 36E curriculum (100ep+50ep) | training schedule | NEG |
| 6 | Exp 34 HN-InfoNCE β-grid | loss shape | NEG (representation-bound) |
| 7 | Exp 37 Stage A ego source | ego compensation | NEG |
| 8 | Exp 37 Stage B OMF 28D | per-cell flow | NEG |
| 9 | Exp 37 Stage C EMAP concat | feature concat | NEG |
| 10 | Exp 37 ORB-grid 3x8 (61D) | per-cell ORB | NEG |
| 11 | Exp 39 CLIP-visual 64D concat | aligner input | NEG (AUC + HOTA) |
| 12 | Exp 40 cliptext aligner | text-side aligner | mixed (iKUN POS, FH NEG) |
| 13 | Exp 41 late-concat motion⊕app | fusion site | NEG |
| 14 | Exp 43 CLIP-logit decision fusion | decision-level | NEG |
| 15 | Case 2 1a fusion transformer | architecture | NEG |
| 16 | Case 2 1b POS-decoupled branches | architecture | NEG |
| 17 | Case 2 1c +ego-state 3rd KV | architecture | NEG |
| 18 | Case 2 1d FiLM on visual | architecture | NEG |
| 19a | **Lever A structural consensus** | aux loss / geometry | **NEG** |
| 19b | **Lever B what/where dual cosine** | score-side text | **NEG** |

Two levers showed honest cross-arch signal but did not unseat ship:
- Depth-augmented 17D (2026-05-10): iKUN +0.215 sig (p=0.016), FH V1/V2 within
  seed noise. iKUN-only candidate; appear-ship multi-seed 44.608 is still the
  conservative reference.
- World-XY metric projection (2026-05-10): FLAT across all 3 archs (aligner
  absorbs unit scale).

## Honest Ceiling Claim

**3-seq pooled HOTA, paper-canonical gt_template_old, Refer-KITTI V1 test
(0005/0011/0013), iKUN architecture, YOLOv8-NS detector:**

| recipe | pooled |
|--------|--------|
| Paper iKUN cascade+simcalib (YOLOv8-NS) | 44.564 |
| Local repro paper-pure | 44.564 |
| GMC-Link iKUN appear-ship (n=3 mean) | **44.608 ± 0.024** |

Δ vs paper = +0.044, one-sided t p=0.044 (significant at α=0.05).

DDETR-NS row 48.84 remains BLOCKED — paper author refused 3× to release tracker
output (iKUN issues #32, #35, #25, #33). NeuralSORT code itself unreleased.
Without DDETR+NS tracker outputs, the 48.84 row is unreachable from any
ceiling-break lever at the iKUN architecture level.

## Infra Kept (Deactivated)

Lever A:
- `StructuralConsensusLoss` class in `gmc_link/losses.py` — generic, may be
  useful for other manifold-alignment tasks; not ship-recipe-compatible.
- `--lam-struct/--lam-angle/--struct-mode` CLI flags in `gmc_link/train.py` —
  default `lam_struct=0.0` (disabled).
- `gmc_link_weights_v1train_levera_s1_l0{1,5}.pth` + corresponding GMC caches
  in `gmc_link/gmc_scores_v1_*_levera_s1_*_cache.json` — reproducibility.
- `gmc_link/manager.py:86-87` ckpt-loading bug fix — unrelated, kept.

Lever B:
- `gmc_link/text_what_where.py` — spaCy POS parser with keyword overrides.
- `iKUN/expr_what_where_v1.json` — 318-expression decomposition map.
- `run_cascade_v1test_with_text_override.py` — generic text-substitution
  cascade driver, useful for future score-side text-rewriting experiments.
- `run_whatwhere_b1_eval.py` — locked-recipe dual-cascade fusion eval.
- `iKUN/ikun_results_v1_cascade_{what,where}.json` — 19 MB caches, kept for
  reproducibility.

## References

- Plan: `/home/seanachan/.claude/plans/valiant-sleeping-mochi.md`
- CDRMOT: arXiv 2503.11496 (Liang et al.)
- DRMOT (reviewed, non-transferable): arXiv 2602.04692
- Lever A memory: `project_lever_a_struct_consensus_negative.md`
- Lever B memory: `project_lever_b_what_where_negative.md`
- Ship reference: `project_ikun_multiseed_positive.md`
- Paper repro: `project_paper_repro_3seq_pooled.md`
- DDETR blocker: `project_ddetr_data_unavailable.md`
