# Path C: LVLM-Rerank as 4th Additive Channel — 2026-05-17

## Goal

Lift GMC-Link iKUN ship from multi-seed n=3 **44.608 ± 0.024** toward 5% target
(~46.84 HOTA). With Paths A (G-DINO detector swap) and DDETR-NS upstream both
blocked, Path C uses an open-source vision-language LVLM (Qwen2-VL) to re-score
ambiguous (track, frame, expression) tuples and add a 4th additive channel atop
the locked iKUN cascade ship recipe.

This is a **score-side lever only**. No detector swap, no tracker swap, no
aligner retrain. The detector (YOLOv8-NS) + tracker (NeuralSORT) +
iKUN cascade-attention + sim_calib + motion+APPEAR GMC channels are all
preserved at the ship configuration that produced the 44.608 multi-seed mean.

## Architecture

Current ship score:
```
ship = cs + b + α_m·(gmc_m − 0.5)·sc_m + α_a·(gmc_a − 0.5)·sc_a
```
where `cs` is cascade cosine, `b` is sim_calib bias, `(α_m, sc_m, thr_m) = (1.0, 0.9, +0.17)`
and `(α_a, sc_a, thr_a) = (1.0, 0.30, +0.10)`.

Path C adds:
```
ship_c = ship + α_l·(lvlm − 0.5)·sc_l
```
where `lvlm ∈ [0, 1]` is a per-(track, frame, expression) score from Qwen2-VL.
Sweep `α_l ∈ {0.0, 0.5, 1.0}`, `sc_l ∈ {1.0, 2.0}`, `thr_l ∈ {0, +0.1, +0.2}` —
27 cells.

## Decomposition into Phases

### Phase C1 — LVLM Score Cache Build + Discrimination Gate (~3 days)

**Files to create:**

- `run_build_lvlm_cache.py` — load `Qwen/Qwen2-VL-2B-Instruct` via HuggingFace
  transformers with **bitsandbytes int4 quantization** (RTX 3060 Ti has 8GB
  total, ~3GB free with neighbor process). 2B int4 = ~1.5GB. If GPU memory
  insufficient even with int4, fallback to InternVL2-1B (~2GB fp16) or
  PaliGemma-3B-mix-224 int4. Iterate over NeuralSORT predict.txt tracks. For
  each (seq, class, expr, fid, track_id) cell, crop bbox from KITTI image,
  build prompt `"Score from 0 to 10 how well this region matches: {expr}.
  Respond with a single integer."`, parse score, divide by 10 → `lvlm ∈ [0, 1]`.
  Output: `lvlm_cache/qwen2vl_v1/{seq}/{class}/{expr}.json` schema
  `{seq, class, expr, model, schema, frames: {fid: {track_id: score}}}`.

  **C1 calibration step**: Before full cache build, run 100-call calibration
  on a single seq/expr to verify (a) prompt produces parseable integer
  responses ≥ 90% of time, (b) GPU memory stable, (c) latency budget
  (≤2s/call on this hardware tier).

- **Cost-bounded filtering** — naïve full eval is ~500K calls × 2s ≈ 12 days
  on Qwen2-VL-7B. Reductions:
  1. Only score (seq, expr) cells where ship cascade leaves frames ambiguous:
     `cs ∈ [thr_cascade − 0.5, thr_cascade + 0.5]`. Cuts ~60% of frames.
  2. Only score detected tracks in expression's class route (car-exprs only
     score car tracks; ped-exprs only score ped tracks). Cuts ~50%.
  3. Use Qwen2-VL-2B first (≈0.3s/call on A100). Realistic budget ~30 hours
     compute for 200K calls.

- Crop strategy: bbox + 20% padding, resize to 448×448 (Qwen2-VL native).
  No SAM2 mask yet (kept as conditional Phase C4 if C2/C3 marginal).

**Gate G1 — Discrimination AUC**

For each (seq, expr) cell with ≥5 positive frames AND ≥5 negative frames in GT:
compute AUC of LVLM scores separating GT-positive from GT-negative tracks at
those frames. Average AUC across cells.

- **PASS G1:** mean AUC ≥ 0.65 across motion-class cells.
- **MARGINAL G1:** mean AUC ∈ [0.55, 0.65] → proceed to C2 with reduced
  expectations (lever may help on subset).
- **FAIL G1:** mean AUC < 0.55 → Qwen2-VL-2B cannot discriminate at task
  granularity. Re-attempt with Qwen2-VL-7B (~3× cost) or kill Path C.

### Phase C2 — 4th Channel Integration + HOTA Gate (~3 days)

**Files to create:**

- `run_ikun_linadd_with_lvlm.py` — fork of `run_ikun_linadd_botsort.py` style
  (apply locked ship + LVLM 4th channel). Load LVLM cache per (seq, class, expr).
  At inference: for each (track, frame, expr) lookup, add α_l·(lvlm−0.5)·sc_l
  to ship score if cell exists in cache (else 0 — frame outside cache range).
- 27-cell sweep over (α_l, sc_l, thr_l), eval 3-seq pooled HOTA per cell.

**Gate G2 — Single-Seed HOTA**

- **PASS G2:** Δship ≥ +0.20 single-seed (any cell in 27-cell grid).
- **MARGINAL G2:** Δship ∈ [+0.05, +0.20] → proceed to C3 to test stability.
- **FAIL G2:** All cells Δship < +0.05 → LVLM signal doesn't survive fusion
  with cascade dot-product. Kill Path C; commit writeup at 44.608.

### Phase C3 — Multi-Seed Validation (~2 days)

Re-run best C2 cell on seeds {1, 2, 3} of the GMC-Link aligner (LVLM cache is
deterministic per fixed crops; seed-variance comes from aligner inference only).

**Gate G3 — Multi-Seed Significance**

- Paired-t (per-seq Δship over 3 V1 seqs × 3 seeds = 9 cells) vs 44.608.
- **PASS G3:** Δ ≥ +0.50 multi-seed mean AND p < 0.10.
- **PARTIAL G3:** Δ ∈ [+0.10, +0.50] → ship as new conservative recipe, then C4.
- **FAIL G3:** Drop lever; commit writeup at 44.608.

### Phase C4 (conditional) — Stack with SAM2 Mask Crops + Qwen2-VL-7B (~2 weeks)

ONLY if C3 PARTIAL but < +2.23 (5% target). Stack:
1. **SAM2 pixel-tight crops** instead of bbox+padding. Tests if LVLM rerank
   loss is bbox-noise dominated.
2. **Qwen2-VL-7B upgrade** from 2B baseline. ~3× compute, ~+5–10% AUC typical.

Each stacked lever has own kill gate (Δ ≥ +0.20 over C3 ship to retain).

## Critical Files

**Phase C1:**
- Create: `run_build_lvlm_cache.py`
- Create: `lvlm_cache/qwen2vl_v1/{seq}/{class}/{expr}.json`
- Create: `run_g1_discrimination_gate.py`
- Update: memory `project_path_c_lvlm_*_{positive,negative}.md`

**Phase C2:**
- Create: `run_ikun_linadd_with_lvlm.py`
- Reuse: ship recipe constants from `run_ikun_linadd_botsort.py`

**Phase C3:**
- Reuse: `run_paired_wilcoxon.py` framework.

## Risks + Mitigations

- **R1: LVLM compute budget overrun.** Realistic worst-case 12 days on full
  set on RTX 3060 Ti. Mitigation: aggressive cascade-ambiguity filter +
  Qwen2-VL-2B int4 quantization (~5× faster than fp16 7B). Budget cap Phase
  C1 at 48 GPU-hours; kill if not complete.

- **R0: GPU memory pressure.** Only ~3GB free on 8GB RTX 3060 Ti. Must
  use int4 quant. If fallback to InternVL2-1B or PaliGemma needed, expect
  weaker discrimination (smaller model → harder for G1 AUC gate).
- **R2: LVLM hallucinates class on crops.** Qwen2-VL trained on natural images,
  KITTI cars are heavily clipped at frame edges. Mitigation: G1 discrimination
  AUC gate catches this cheaply.
- **R3: 4th channel collinear with cascade.** If LVLM mostly re-encodes the
  same evidence cascade already has, additive channel adds noise. Mitigation:
  G2 sweep covers α_l=0 as control; if best α_l > 0 sig at G2 we confirm
  independence.
- **R4: Score scale mismatch on integer parse.** "0–10" prompt has discrete
  output; LVLM may saturate at 5 or 10. Mitigation: histogram check after
  C1 build, fall back to logit-extraction or cosine-sim of LVLM hidden state
  if too quantized.
- **R5: 5% target unreachable from single lever.** Honest scope: Path C base
  (no SAM2, no 7B) realistic range +0.0 to +0.5 HOTA. C4 stack +0.5 to +1.5.
  Full 5% (+2.23) unlikely from Path C alone — would need C4 + additional
  Path D (TBD: re-rank loss-distillation onto aligner?). Spec budgets
  for honest +0.5 ship.

## Decision Tree

```
C1: build LVLM cache → G1 discrimination
  G1 FAIL (AUC<0.55) → 7B retry; if still FAIL → kill, writeup
  G1 MARGINAL → C2 with reduced expectations
  G1 PASS → C2
C2: 4th channel sweep → G2 HOTA
  G2 FAIL (Δ<+0.05) → kill, writeup
  G2 MARGINAL → C3
  G2 PASS → C3
C3: multi-seed → G3 significance
  G3 FAIL → writeup
  G3 PARTIAL → ship + C4 if 5% still target
  G3 PASS → ship; C4 only if 5% required
C4: SAM2 + 7B stack → final ship
```

Wall-clock cap: **4 weeks total** across C1+C2+C3+C4. Exceed → ship whatever
stack is alive at C3.

## Verification

1. **G1 discrimination AUC report** per-(seq, expr) cell, saved to
   `diagnostics/results/lvlm/discrimination_auc.{json,md}`.
2. **Score-histogram regression check** per-class in C1: detect quantization
   saturation (>50% scores at one integer value).
3. **G2 HOTA report** via standard TrackEval, save 27-cell grid table.
4. **G3 paired-t** reuse `run_paired_wilcoxon.py` format.
5. **Cache-coverage report**: per-(seq, expr) frame coverage % vs full
   trajectory — confirm cascade-ambiguity filter didn't drop ground-truth
   positive frames disproportionately (bias check).

## Final Write-up Trigger

Commit to writeup at whatever C3 or C4 delivers if:
- (G1 FAIL ∨ G2 FAIL ∨ G3 FAIL), OR
- Wall-clock exceeds 4 weeks, OR
- Multi-seed mean ≥ +2.23 (5% target hit) → ship + paper update.

Whichever first.

## Open Questions

1. **Prompt engineering**: integer 0–10 vs Yes/No vs continuous logit? Decide
   in Phase C1 after a quick 100-call calibration sweep.
2. **Cache schema**: per-expr file vs per-class consolidated file. Per-expr
   easier to extend; per-class lower fileops overhead.
3. **Qwen2-VL-7B vs 2B starting point**: Spec defaults 2B for cost. If G1 FAIL
   on 2B, retry on 7B before killing.
4. **Whether to test Path D (LVLM logit distillation into aligner) atop C**:
   deferred — only spec if C3 PARTIAL but <5%.
