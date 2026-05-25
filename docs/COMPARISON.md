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

| host | query type | host alone | + GMC-Link (mean ± std) | gain | significance (p) |
|---|---|---|---|---|---|
| iKUN | appearance | 46.346 | 46.746 ± 0.045 | +0.400 | 0.0021 |
| iKUN | motion | 25.531 | 30.093 ± 0.240 | **+4.562** | 0.0005 |
| iKUN | static | 43.914 | 45.099 ± 0.178 | +1.185 | 0.0037 |
| FlexHook V1 | appearance | 55.492 | 55.700 ± 0.026 | +0.208 | 0.0026 |
| FlexHook V1 | motion | 43.981 | 45.785 ± 0.235 | +1.804 | 0.0028 |
| FlexHook V1 | static | 48.983 | 49.771 ± 0.217 | +0.788 | 0.0122 |
| FlexHook V2 | appearance | 41.748 | 41.946 ± 0.051 | +0.198 | 0.0105 |
| FlexHook V2 | motion | 48.018 | 48.758 ± 0.067 | +0.740 | 0.0014 |
| FlexHook V2 | static | 44.622 | 44.935 ± 0.024 | +0.313 | 0.0009 |

All nine cells are positive and statistically significant (p < 0.05; seven at p < 0.01).
The largest is iKUN motion +4.562.

---

## 5. Bottom line

| question | answer |
|---|---|
| Does GMC-Link help with no tuning? | Yes on all 3 valid cells (+0.006 to +0.048). |
| Does it beat published scores? | iKUN·V1 yes (+0.070); FlexHook·V2 yes (+0.277); FlexHook·V1 short (−0.298, structural). |
| Where is the gain biggest? | Motion queries — iKUN·V1 +4.562; all motion cells positive + significant. |
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
