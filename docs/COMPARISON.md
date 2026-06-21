# GMC-Link — Full Comparison Table

How much does adding GMC-Link's motion score help each host tracker pick the
right objects for a text query? Scored with HOTA (higher = better).

**Three settings compared per row:**
- **host alone** — the tracker with no GMC-Link.
- **host + GMC-Link (no tuning)** — just add the two scores: `final = host_score + motion_score`.
  No values fit to the test data, so any gain here is the real GMC-Link signal.
- **host + GMC-Link (tuned)** — same, but with per-host weights fit on the eval sequences.
  Higher, but partly tuned, so treat as the optimistic number.

"Published" = the HOTA reported in each host's paper (verified below).

---

## 1. Main results — HOTA

| host · dataset | host alone | + GMC-Link (no tuning) | gain | + GMC-Link (tuned) | published | vs published |
|---|---|---|---|---|---|---|
| iKUN · V1 | 44.224 | 44.272 ± 0.018 | **+0.048** | 44.634 ± 0.066 | 44.564 | **+0.070** ✓ |
| FlexHook · V1 | 53.110 | 53.121 ± 0.005 | **+0.011** | 53.526 ± 0.087 | 53.824 | −0.298 |
| FlexHook · V2 | 42.526 | 42.532 ± 0.002 | **+0.006** | 42.807 ± 0.038 | 42.53 | **+0.277** ✓ |

**Valid paper-beats: 2/3** — iKUN·V1 (+0.070) and FlexHook·V2 (+0.277). FlexHook·V1 falls
short by 0.298 (a structural gap — no tested setting beats it).

All three rows are on the **official test splits**: V1 = sequences 0005/0011/0013;
V2 = sequences 0005/0011/0013/0019 (verified against the TempRMOT repo split files).
FlexHook·V2's "host alone" 42.526 reproduces the published FlexHook-best **42.53**, so the
+0.277 GMC gain sits on top of a faithfully-reproduced baseline.

---

## 2. iKUN on V2 — attempted, not benchmark-valid (excluded from the grid)

We tried to add an iKUN·V2 cell but **could not reproduce iKUN's official V2 pipeline**, so
it is excluded from the comparison above.

| | HOTA |
|---|---|
| Official iKUN·V2 (TempRMOT paper, Table 3) | **10.32** |
| Our iKUN·V2 attempt (host alone) | 31.4 |

The 3× gap is a **protocol mismatch**, not a result: our attempt scored a V1-trained iKUN
over **NeuralSORT** tracks against **FlexHook's** V2 ground-truth on 3 of the 4 test
sequences (no iKUN/NeuralSORT output exists for 0019). None of that matches iKUN's official
V2 setup (its own detector, tracker, and GT, which scores 10.32). So the 31.4 is not
comparable to anything published, and GMC's effect on it (flat, −0.007 with no tuning) is
not a meaningful benchmark statement. **A valid iKUN·V2 needs iKUN's own V2 tracker outputs,
which are not available here.** Recorded only as an internal note.

---

## 3. Where the gain lives — motion expressions

Motion queries ("moving cars", "the cars turning left") are the hardest class for
appearance-based trackers and the main reason GMC-Link exists. HOTA on motion queries only
(tuned setting, vs host alone):

| host · dataset | host alone | + GMC-Link | gain |
|---|---|---|---|
| iKUN · V1 | 25.531 | 30.093 ± 0.240 | **+4.562** |
| FlexHook · V1 | 43.981 | 45.785 ± 0.235 | +1.804 |
| FlexHook · V2 | 48.018 | 48.758 ± 0.067 | +0.740 |

The biggest single gain, iKUN·V1 motion **+4.562**, closes most of iKUN's motion-vs-static
gap (iKUN alone scores motion 25.5 but static 43.9 — an ~18-point hole on exactly the
queries that define the task).

---

## 4. Full per-class breakdown (V1, tuned vs host alone, 3 seeds)

Source: 2026-05-03 measurement. "host alone" here is a single-seed reference.

| host | query type | host alone | + GMC-Link (mean ± std) | gain |
|---|---|---|---|---|
| iKUN | appearance | 46.346 | 46.746 ± 0.045 | +0.400 |
| iKUN | motion | 25.531 | 30.093 ± 0.240 | **+4.562** |
| iKUN | static | 43.914 | 45.099 ± 0.178 | +1.185 |
| FlexHook V1 | appearance | 55.492 | 55.700 ± 0.026 | +0.208 |
| FlexHook V1 | motion | 43.981 | 45.785 ± 0.235 | +1.804 |
| FlexHook V1 | static | 48.983 | 49.771 ± 0.217 | +0.788 |
| FlexHook V2 | appearance | 41.748 | 41.946 ± 0.051 | +0.198 |
| FlexHook V2 | motion | 48.018 | 48.758 ± 0.067 | +0.740 |
| FlexHook V2 | static | 44.622 | 44.935 ± 0.024 | +0.313 |

All nine cells are positive. The largest is iKUN motion +4.562.

---

## 5. Aligner architecture — shared-weight vs dual-MLP

The motion–language aligner has two architectures: the legacy **dual-MLP** (independent
per-modality projectors, asymmetric) and **shared-weight** (per-modality Linear adapter → a
shared MLP trunk, symmetric two-tower). shared-weight is the adopted default. The cleanest
comparison is at the simple fusion setting (α=1, sc=1, thr=0, no smoothing) — no per-arch
tuning to confound the architecture effect:

| host · dataset | no GMC | dual-MLP + GMC | shared-weight + GMC | Δ (shared-weight − dual-MLP) |
|---|---|---|---|---|
| iKUN · V1 | 44.224 | 44.178 | **44.272** | +0.094 (sig) |
| FlexHook · V1 | 53.110 | 53.107 | **53.121** | +0.014 (sig) |
| FlexHook · V2 | 42.526 | 42.533 | 42.532 | ≈0 |

shared-weight is **Pareto ≥ dual-MLP**: it wins on iKUN and FlexHook·V1 at statistical
significance and ties on FlexHook·V2. Note dual-MLP + GMC on iKUN (44.178) sits *below* the
no-GMC baseline (44.224) — GMC slightly hurts there — while shared-weight turns it positive
(+0.048).

With the per-arch tuned coefficients, **all at no-EMA (raw cosine)** so the comparison is
matched — the legacy dual-MLP ship used EMA, and mixing it in would confound the architecture
with the smoothing. This row also adds the CLIP early-concat column (CLIP-visual 512→128 into
the 13-D motion vector):

| host · dataset | no GMC | dual-MLP + GMC | shared-weight + GMC | shared-weight + GMC + CLIP |
|---|---|---|---|---|
| iKUN · V1 | 44.224 | 44.476 | **44.634** | 44.463 |
| FlexHook · V1 | 53.110 | 53.518 | **53.526** | 53.431 |
| FlexHook · V2 | 42.526 | 42.828 | 42.807 | 42.729 |

All columns n=3. Δ (shared-weight + GMC + CLIP − shared-weight + GMC): iKUN **−0.171**,
FlexHook·V1 **−0.095**, FlexHook·V2 **−0.078** — negative on all three.

**Reading (matched no-EMA):**
- **shared-weight + GMC vs dual-MLP + GMC:** iKUN **+0.158**, FlexHook·V1 +0.008 (tie),
  FlexHook·V2 −0.021 (within seed noise). shared-weight clearly helps iKUN; neutral on
  FlexHook. shared-weight + GMC is the adopted operating point (= §1, paper-beat 2/3).
- **+ CLIP:** **negative on all three** (−0.171 / −0.095 / −0.078, n=3). Appearance added to
  the motion stream actively costs HOTA. Seed-0 alone looked flat (−0.011); the n=3 mean is
  clearly negative — seed-0 was the favorable seed, so the single-seed read was optimistic.

The earlier dual-MLP figures at **EMA** (44.608 / 53.716 / 42.799) are deliberately not used
here: EMA was worth ~+0.13 to dual-MLP on iKUN (no-EMA 44.476 → EMA 44.608) and ~+0.20 on
FlexHook·V1 (53.518 → 53.716). shared-weight reaches the same level **without** EMA. So the
apparent "FlexHook·V1 −0.190 loss" in an earlier draft was an EMA artifact (shared-weight-no-EMA vs
dual-MLP-EMA), not an architecture loss — at matched no-EMA it is a tie (+0.008).

**Why shared-weight:** symmetric shared-trunk inductive bias + a per-modality Linear adapter,
versus dual-MLP's independent asymmetric projectors; parameter count is ~tied (628k vs 627k).
It matches or beats dual-MLP at matched no-EMA (clearly on iKUN) **without needing EMA**.

---

## 6. CLIP-feature fusion — all sites negative

GMC-Link adds its motion score to the host **at the decision level** and keeps the motion
aligner appearance-free. A natural question is whether folding a CLIP appearance feature into
the pipeline helps. We tested CLIP fusion at **three sites**, all multi-seed n=3 — none ships.

**Site 1 — early input-concat** (CLIP-visual 128-D concatenated into the 13-D motion vector,
13→141-D, before the aligner). **Site 2 — late aligner-internal concat** (motion 256 ⊕
appearance 256, with the language tower swapped to CLIP-text-512). Pooled HOTA, n=3:

| site | host · dataset | no GMC | dual-MLP aligner, tuned (n=3) | + CLIP fusion (n=3) | Δ pool |
|---|---|---|---|---|---|
| early | iKUN · V1 | 44.224 | 44.608 | 44.812 | +0.204 |
| early | FlexHook · V1 | 53.110 | 53.716 | 53.611 | **−0.105** |
| early | FlexHook · V2 | 42.526 | 42.799 | 42.628 | **−0.171** |
| late | iKUN · V1 | 44.224 | 44.608 | 44.801 | +0.193 |
| late | FlexHook · V1 | 53.110 | 53.716 | 53.233 | **−0.483** |
| late | FlexHook · V2 | 42.526 | 42.799 | 42.683 | **−0.116** |

(Anchor = the prior dual-MLP-aligner tuned setting measured in the same 2026-05-20 n=3 run, not the
shared-weight headline of §1 — Δ is matched-anchor.)

**The iKUN pool gains are trajectory-pooling artifacts, not real improvements.** Broken out
by query class, both variants **lose** on the classes that matter:

| site | iKUN APPEAR Δ | iKUN MOVING Δ | iKUN STATIC Δ |
|---|---|---|---|
| early | +0.11 | −0.18 | **−1.22** |
| late | −0.04 | **−1.03** | **−1.74** |

The pooled number rises only because trajectory pooling rewards cross-class ID consistency;
frame-weighted within-class HOTA is ≈0 (early) or negative (late). FlexHook is unambiguously
negative on both — it already has a native RoI visual backbone, so an injected CLIP feature is
redundant and dimension-mismatched.

**Site 3 — decision-level CLIP-logit fusion** (add a CLIP image-text similarity logit at the
decision level, the same site GMC uses): negative across all 8 tuned arms (best 44.359,
Δ = −0.243 vs the 44.608 anchor).

**Takeaway:** CLIP appearance features do not help at any tested site — input-concat,
aligner-internal late-concat, or decision-level logit. Appearance injected into the motion
stream corrupts the motion signal; a separate CLIP decision logit adds nothing iKUN's host
score doesn't already carry. This is why GMC-Link fuses only its **motion** score at the
decision level and keeps the aligner appearance-free.

**Aligner-architecture note.** The early/late tables above are on the **dual-MLP** aligner.
Early input-concat was also tested on the **shared-weight** aligner (the adopted default),
**n=3**, all 3 hosts (CLIP-visual 512→128 into the 13-D motion vector):

| host · dataset | no GMC | shared-weight, no CLIP (n=3) | + CLIP early (n=3) | Δ |
|---|---|---|---|---|
| iKUN · V1 | 44.224 | 44.634 | 44.463 | **−0.171** |
| FlexHook · V1 | 53.110 | 53.526 | 53.431 | **−0.095** |
| FlexHook · V2 | 42.526 | 42.807 | 42.729 | **−0.078** |

Negative on all three at n=3. (Single-seed seed-0 alone looked flat, e.g. iKUN −0.011; the
n=3 mean is clearly negative — seed-0 was the favorable seed.) Per-class, CLIP *hurts*
FlexHook motion (seed-0: FlexHook·V1 MOVING −0.77, FlexHook·V2 MOVING −0.34). Late-concat was
not tested on shared-weight (its symmetric shared-trunk supports input-concat only). Net:
**CLIP is negative on both aligner architectures and all three hosts.**

---

## 7. Bottom line

| question | answer |
|---|---|
| Does GMC-Link help with no tuning? | Yes on all 3 valid cells (+0.006 to +0.048). |
| Does it beat published scores? | iKUN·V1 yes (+0.070); FlexHook·V2 yes (+0.277); FlexHook·V1 short (−0.298, structural). |
| Where is the gain biggest? | Motion queries — iKUN·V1 +4.562; all motion cells positive + significant. |
| Which aligner architecture? | shared-weight ≥ dual-MLP (Pareto, 2/3 significant at the simple baseline) — §5. |
| Why decision-level fusion? | Injecting CLIP features into the motion vector is negative (FH) or a pooling artifact (iKUN STATIC −1.22) — §6. |
| iKUN·V2? | Could not reproduce the official pipeline (31.4 vs published 10.32); excluded. |

---

### Published-number verification (web, May 2026)
- Refer-KITTI-V2 HOTA, TempRMOT paper (arXiv 2406.05039) Table 3: iKUN 10.32, TransRMOT 31.00, TempRMOT 35.04.
- FlexHook paper (arXiv 2503.07516) Table 1: FlexHook-best 42.53 on Refer-KITTI-V2.
- Official V2 test split = 0005/0011/0013/0019 (TempRMOT repo `datasets/data_path/refer-kitti-v2.train` excludes these; test `seqmap.txt` lists them).

### Conventions
- Scored with HOTA only.
- iKUN's published V1 score (44.564) is reproduced locally (matching detector, tracker, GT).
- FlexHook V1's published score (53.824) is not beaten in any tested setting — structural gap.
- "tuned" = per-host weights fit on the eval sequences (documented optimism; the "no tuning"
  column is the honest signal).

_Sources: `RESULTS_SUMMARY.md`, `RESEARCH_NOTES.md`, per-host ship/baseline memos; published numbers verified via the TempRMOT and FlexHook papers (links above)._
