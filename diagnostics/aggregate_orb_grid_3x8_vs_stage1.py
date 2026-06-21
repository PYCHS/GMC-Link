#!/usr/bin/env python3
"""Compare zoned-orb 3x8 (61D) vs production stage1 (13D) aligners.

Reads per-seq layer3 npz files for both models on seqs 0005/0011/0013, computes
per-expression similarity-based micro-aggregated metrics across seqs, and writes
side-by-side delta tables to a Markdown file.

Δ convention: orb 3x8 − stage1.  Positive Δ AUC → orb 3x8 helped.

Output:
  diagnostics/results/multiseq/comparison_orb_grid_3x8_vs_stage1.md
  diagnostics/results/multiseq/comparison_orb_grid_3x8_vs_stage1.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu


RESULTS_DIR = Path("/home/seanachan/GMC-Link/diagnostics/results/exp37")
OUTPUT_DIR = Path("/home/seanachan/GMC-Link/diagnostics/results/multiseq")
SEQS = ["0005", "0011", "0013"]
MODELS = {
    "stage1":  "v1train_stage1",          # production 13D baseline
    "orb3x8":  "v1train_orb_grid_3x8",    # 61D (13D + 48D zoned-orb 3x8)
}
NEW_LABEL = "orb 3x8 (61D)"
BASE_LABEL = "stage1 (13D)"
NEW_KEY = "orb3x8"
BASE_KEY = "stage1"
OUT_NAME = "comparison_orb_grid_3x8_vs_stage1"


def load_seq_arrays(model_tag: str, seq: str):
    path = RESULTS_DIR / f"layer3_{seq}_{model_tag}.npz"
    d = np.load(path, allow_pickle=True)
    results = d["results"].tolist()
    gt_list = d["gt_cosines_by_expr"]
    nongt_list = d["nongt_cosines_by_expr"]
    out = {}
    for i, r in enumerate(results):
        sent = r["sentence"]
        out[sent] = (
            np.asarray(gt_list[i], dtype=np.float64),
            np.asarray(nongt_list[i], dtype=np.float64),
        )
    return out


def pool_across_seqs(model_tag: str, seqs):
    per_seq = {s: load_seq_arrays(model_tag, s) for s in seqs}
    sentences = sorted({sent for d in per_seq.values() for sent in d.keys()})
    pooled = {}
    for sent in sentences:
        gt_parts = []
        nongt_parts = []
        for s in seqs:
            arr = per_seq[s].get(sent)
            if arr is None:
                continue
            gt, nongt = arr
            if gt.size > 0:
                gt_parts.append(gt)
            if nongt.size > 0:
                nongt_parts.append(nongt)
        gt_pooled = (
            np.concatenate(gt_parts) if gt_parts else np.array([], dtype=np.float64)
        )
        nongt_pooled = (
            np.concatenate(nongt_parts) if nongt_parts else np.array([], dtype=np.float64)
        )
        pooled[sent] = (gt_pooled, nongt_pooled)
    return pooled


def compute_metrics(gt: np.ndarray, nongt: np.ndarray) -> dict:
    n_gt = int(gt.size)
    n_nongt = int(nongt.size)
    if n_gt == 0 or n_nongt == 0:
        return dict(
            n_gt=n_gt, n_nongt=n_nongt,
            auc=None, sep=None, gt_mean=None, nongt_mean=None,
            gt_std=None, nongt_std=None, cohen_d=None,
        )
    gt_mean = float(gt.mean())
    nongt_mean = float(nongt.mean())
    gt_std = float(gt.std(ddof=0))
    nongt_std = float(nongt.std(ddof=0))
    sep = gt_mean - nongt_mean
    pooled_var = (gt_std ** 2 + nongt_std ** 2) / 2.0
    cohen_d = sep / np.sqrt(pooled_var) if pooled_var > 0 else 0.0
    try:
        U, _ = mannwhitneyu(gt, nongt, alternative="greater")
        auc = float(U / (n_gt * n_nongt))
    except ValueError:
        auc = 0.5
    return dict(
        n_gt=n_gt, n_nongt=n_nongt,
        auc=auc, sep=float(sep),
        gt_mean=gt_mean, nongt_mean=nongt_mean,
        gt_std=gt_std, nongt_std=nongt_std,
        cohen_d=float(cohen_d),
    )


def overall_micro(model_pooled) -> dict:
    gt_all = (
        np.concatenate([v[0] for v in model_pooled.values() if v[0].size > 0])
        if any(v[0].size for v in model_pooled.values()) else np.array([])
    )
    nongt_all = (
        np.concatenate([v[1] for v in model_pooled.values() if v[1].size > 0])
        if any(v[1].size for v in model_pooled.values()) else np.array([])
    )
    return compute_metrics(gt_all, nongt_all)


def fmt(x, d=3):
    if x is None:
        return "—"
    return f"{x:.{d}f}"


def fmt_delta(x, d=3):
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{d}f}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pooled arrays for both models across seqs {SEQS}...")
    pooled = {tag: pool_across_seqs(MODELS[tag], SEQS) for tag in MODELS}

    expressions = sorted(set(pooled[BASE_KEY].keys()) | set(pooled[NEW_KEY].keys()))
    rows = []
    for sent in expressions:
        m_b = compute_metrics(*pooled[BASE_KEY].get(sent, (np.array([]), np.array([]))))
        m_n = compute_metrics(*pooled[NEW_KEY].get(sent, (np.array([]), np.array([]))))
        delta = {}
        for k in ("auc", "sep", "gt_mean", "nongt_mean", "cohen_d"):
            if m_b[k] is None or m_n[k] is None:
                delta[k] = None
            else:
                delta[k] = m_n[k] - m_b[k]
        rows.append({
            "sentence": sent,
            "base": m_b,
            "new":  m_n,
            "delta":  delta,
        })

    base_overall = overall_micro(pooled[BASE_KEY])
    new_overall  = overall_micro(pooled[NEW_KEY])
    delta_overall = {}
    for k in ("auc", "sep", "gt_mean", "nongt_mean", "cohen_d"):
        if base_overall[k] is None or new_overall[k] is None:
            delta_overall[k] = None
        else:
            delta_overall[k] = new_overall[k] - base_overall[k]

    def sort_key(r):
        d = r["delta"]["auc"]
        return (-abs(d) if d is not None else 1.0, r["sentence"])
    rows.sort(key=sort_key)

    inversions = []
    for r in rows:
        d_b = r["base"]["cohen_d"]
        d_n = r["new"]["cohen_d"]
        if d_b is None or d_n is None:
            continue
        if (d_b >= 0) != (d_n >= 0):
            inversions.append((r["sentence"], d_b, d_n))

    valid_rows = [r for r in rows if r["delta"]["auc"] is not None]
    biggest_pos = sorted(valid_rows, key=lambda r: -r["delta"]["auc"])[:5]
    biggest_neg = sorted(valid_rows, key=lambda r: r["delta"]["auc"])[:5]

    n_helped = sum(1 for r in valid_rows if r["delta"]["auc"] > 0.001)
    n_hurt   = sum(1 for r in valid_rows if r["delta"]["auc"] < -0.001)
    n_flat   = len(valid_rows) - n_helped - n_hurt

    lines = []
    lines.append(f"# Zoned-orb {NEW_LABEL} vs production {BASE_LABEL}")
    lines.append("")
    lines.append("**Held-out seqs**: " + ", ".join(SEQS))
    lines.append(f"**n expressions**: {len(valid_rows)}")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"Δ = {NEW_LABEL} − {BASE_LABEL}.  Positive Δ AUC → orb 3x8 zoned-flow concat HELPED, "
        f"negative → it HURT vs the {BASE_LABEL} production stage1."
    )
    lines.append("")
    lines.append(
        f"Of {len(valid_rows)} expressions, **{n_helped} were helped** by orb 3x8 zoned flow "
        f"(Δ > +0.001), **{n_hurt} were hurt** (Δ < −0.001), **{n_flat} stayed flat**. "
        f"Overall micro AUC moved from **{fmt(base_overall['auc'])}** ({BASE_LABEL}) to "
        f"**{fmt(new_overall['auc'])}** ({NEW_LABEL}), Δ = {fmt_delta(delta_overall['auc'])}."
    )
    lines.append("")
    if inversions:
        lines.append(
            f"**{len(inversions)} expressions had a Cohen's d sign flip** "
            "(see Inversions section)."
        )
    else:
        lines.append("**No expressions had a Cohen's d sign flip** — "
                     "discrimination direction preserved.")
    lines.append("")

    lines.append("## Summary (overall, pooled across all expressions and seqs)")
    lines.append("")
    lines.append(f"| metric | {BASE_LABEL} | {NEW_LABEL} | Δ ({NEW_LABEL} − {BASE_LABEL}) |")
    lines.append("|---|---|---|---|")
    for k, label in [
        ("auc", "AUC (micro)"),
        ("sep", "separation"),
        ("gt_mean", "GT mean"),
        ("nongt_mean", "non-GT mean"),
        ("cohen_d", "Cohen d"),
    ]:
        lines.append(
            f"| {label} | {fmt(base_overall[k], 4)} | "
            f"{fmt(new_overall[k], 4)} | "
            f"{fmt_delta(delta_overall[k], 4)} |"
        )
    lines.append("")
    lines.append(
        f"**Sample counts (pooled)**: "
        f"{BASE_LABEL} gt={base_overall['n_gt']}, ngt={base_overall['n_nongt']}; "
        f"{NEW_LABEL} gt={new_overall['n_gt']}, ngt={new_overall['n_nongt']}."
    )
    lines.append("")

    lines.append(f"## Top 5 expressions where {NEW_LABEL} HELPED (most positive Δ AUC)")
    lines.append("")
    lines.append(f"| Expression | {BASE_LABEL} AUC | {NEW_LABEL} AUC | Δ AUC | base d | new d | Δ d |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in biggest_pos:
        lines.append(
            f"| {r['sentence']} | "
            f"{fmt(r['base']['auc'])} | "
            f"{fmt(r['new']['auc'])} | "
            f"{fmt_delta(r['delta']['auc'])} | "
            f"{fmt(r['base']['cohen_d'])} | "
            f"{fmt(r['new']['cohen_d'])} | "
            f"{fmt_delta(r['delta']['cohen_d'])} |"
        )
    lines.append("")

    lines.append(f"## Top 5 expressions where {NEW_LABEL} HURT (most negative Δ AUC)")
    lines.append("")
    lines.append(f"| Expression | {BASE_LABEL} AUC | {NEW_LABEL} AUC | Δ AUC | base d | new d | Δ d |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in biggest_neg:
        lines.append(
            f"| {r['sentence']} | "
            f"{fmt(r['base']['auc'])} | "
            f"{fmt(r['new']['auc'])} | "
            f"{fmt_delta(r['delta']['auc'])} | "
            f"{fmt(r['base']['cohen_d'])} | "
            f"{fmt(r['new']['cohen_d'])} | "
            f"{fmt_delta(r['delta']['cohen_d'])} |"
        )
    lines.append("")

    if inversions:
        lines.append("## Cohen's d sign flips (inversions)")
        lines.append("")
        lines.append(f"| Expression | {BASE_LABEL} d | {NEW_LABEL} d |")
        lines.append("|---|---|---|")
        for sent, d_b, d_n in inversions:
            lines.append(f"| {sent} | {fmt(d_b)} | {fmt(d_n)} |")
        lines.append("")

    lines.append("## Per-expression delta (sorted by |Δ AUC| descending)")
    lines.append("")
    lines.append(
        f"| Expression | {BASE_LABEL} AUC | {NEW_LABEL} AUC | Δ AUC | base sep | new sep | Δ sep | base d | new d | Δ d |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['sentence']} | "
            f"{fmt(r['base']['auc'])} | "
            f"{fmt(r['new']['auc'])} | "
            f"{fmt_delta(r['delta']['auc'])} | "
            f"{fmt(r['base']['sep'])} | "
            f"{fmt(r['new']['sep'])} | "
            f"{fmt_delta(r['delta']['sep'])} | "
            f"{fmt(r['base']['cohen_d'])} | "
            f"{fmt(r['new']['cohen_d'])} | "
            f"{fmt_delta(r['delta']['cohen_d'])} |"
        )
    lines.append("")

    lines.append("## Per-expression GT vs non-GT means")
    lines.append("")
    lines.append(
        f"| Expression | base GT μ | new GT μ | Δ GT μ | base ngt μ | new ngt μ | Δ ngt μ | n_gt | n_ngt |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        n_gt = r["base"]["n_gt"]
        n_ngt = r["base"]["n_nongt"]
        lines.append(
            f"| {r['sentence']} | "
            f"{fmt(r['base']['gt_mean'])} | "
            f"{fmt(r['new']['gt_mean'])} | "
            f"{fmt_delta(r['delta']['gt_mean'])} | "
            f"{fmt(r['base']['nongt_mean'])} | "
            f"{fmt(r['new']['nongt_mean'])} | "
            f"{fmt_delta(r['delta']['nongt_mean'])} | "
            f"{n_gt} | {n_ngt} |"
        )
    lines.append("")

    out_md = OUTPUT_DIR / f"{OUT_NAME}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")

    out_json = OUTPUT_DIR / f"{OUT_NAME}.json"
    payload = {
        "seqs": SEQS,
        "headline": {
            BASE_KEY: base_overall,
            NEW_KEY:  new_overall,
            f"delta_{NEW_KEY}_minus_{BASE_KEY}": delta_overall,
        },
        "per_expression": [
            {
                "sentence": r["sentence"],
                BASE_KEY: r["base"],
                NEW_KEY:  r["new"],
                f"delta_{NEW_KEY}_minus_{BASE_KEY}": r["delta"],
            }
            for r in rows
        ],
        "inversions": [
            {"sentence": s, f"{BASE_KEY}_cohen_d": db, f"{NEW_KEY}_cohen_d": dn}
            for s, db, dn in inversions
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_json}")

    print()
    print("=" * 80)
    print("HEADLINE")
    print("=" * 80)
    print(f"  {BASE_LABEL}: AUC={fmt(base_overall['auc'], 4)}  "
          f"sep={fmt(base_overall['sep'], 4)}  d={fmt(base_overall['cohen_d'], 3)}")
    print(f"  {NEW_LABEL}: AUC={fmt(new_overall['auc'], 4)}  "
          f"sep={fmt(new_overall['sep'], 4)}  d={fmt(new_overall['cohen_d'], 3)}")
    print(f"  Δ ({NEW_LABEL} − {BASE_LABEL}): "
          f"AUC={fmt_delta(delta_overall['auc'], 4)}  "
          f"sep={fmt_delta(delta_overall['sep'], 4)}  "
          f"d={fmt_delta(delta_overall['cohen_d'], 3)}")
    print()
    print(f"Top 5 {NEW_LABEL} HELPED (Δ AUC most positive):")
    for r in biggest_pos:
        print(f"  {fmt_delta(r['delta']['auc'])}  {r['sentence']}")
    print()
    print(f"Top 5 {NEW_LABEL} HURT (Δ AUC most negative):")
    for r in biggest_neg:
        print(f"  {fmt_delta(r['delta']['auc'])}  {r['sentence']}")
    print()
    if inversions:
        print(f"Cohen's d sign flips ({len(inversions)}):")
        for sent, db, dn in inversions:
            print(f"  {sent}: base d={fmt(db)}, new d={fmt(dn)}")
    else:
        print("No Cohen's d sign flips.")


if __name__ == "__main__":
    main()
