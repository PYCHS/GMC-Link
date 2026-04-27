#!/usr/bin/env python3
"""Evaluate the learned posthoc state-aware gate against:
    1. Raw stage1 cosine
    2. Analytical gate (alpha=0.5, sigma=4.0) from posthoc_gate_mvp

All metrics are SEPARATION (sep = mean(GT cos) - mean(non-GT cos)) on V1
holdout (0005, 0011, 0013).

Run:
    ~/miniconda/envs/RMOT/bin/python eval_learned_gate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gmc_link.learned_state_gate import LearnedStateGate, expr_class_to_onehot
from gmc_link.text_utils import TextEncoder
from run_posthoc_state_gate_mvp import collect_raw_cosines_for_seq, classify_expr


HOLDOUT_SEQS = ["0005", "0011", "0013"]
DEFAULT_WEIGHTS = "gmc_link_weights_v1train_stage1.pth"
DEFAULT_GATE = "learned_gate_v1train.pt"
ANALYTICAL_ALPHA = 0.5
ANALYTICAL_SIGMA = 4.0


def load_gate(gate_path: str, device: torch.device) -> LearnedStateGate:
    ckpt = torch.load(gate_path, map_location=device, weights_only=False)
    gate = LearnedStateGate(
        lang_dim=ckpt.get("lang_dim", 384),
        delta_bound=ckpt.get("delta_bound", 0.5),
        sigma_default=ckpt.get("sigma_default", 4.0),
    ).to(device)
    gate.load_state_dict(ckpt["model"])
    gate.eval()
    return gate


def evaluate_all(weights: str, gate_path: str, seqs: List[str], device: torch.device):
    """Compute pooled per-expression sep for raw / analytical / learned.

    Returns: dict with per_expr list and pooled summary.
    """
    gate = load_gate(gate_path, device)
    encoder = TextEncoder(model_name="all-MiniLM-L6-v2", device=str(device))

    # Pooled accumulators per sentence
    pooled = defaultdict(lambda: {
        "expr_class": None,
        "raw_gt": [], "raw_ngt": [],
        "ana_gt": [], "ana_ngt": [],
        "lrn_gt": [], "lrn_ngt": [],
    })

    # Cache embeddings + onehots per sentence to avoid recomputation
    sent_emb: Dict[str, torch.Tensor] = {}
    sent_oh: Dict[str, torch.Tensor] = {}

    for seq in seqs:
        print(f"\n=== Eval seq {seq} ===")
        seq_data = collect_raw_cosines_for_seq(weights, seq, device)
        if seq_data is None:
            print(f"[skip] seq={seq}")
            continue
        d_track = seq_data["d_track"]
        for expr_rec in seq_data["expressions"]:
            sentence = expr_rec["sentence"]
            klass = expr_rec["expr_class"]
            gt_tids = expr_rec["gt_tids"]
            pooled[sentence]["expr_class"] = klass

            if sentence not in sent_emb:
                sent_emb[sentence] = encoder.encode(sentence).detach().cpu().squeeze(0).float().to(device)
                sent_oh[sentence] = expr_class_to_onehot(klass).to(device)

            emb = sent_emb[sentence]
            oh = sent_oh[sentence]

            for tid, cos_arr in expr_rec["track_cos"].items():
                if tid not in d_track:
                    continue
                d = float(d_track[tid])
                n = len(cos_arr)
                if n == 0:
                    continue

                # Analytical gate
                s = float(np.exp(-d / ANALYTICAL_SIGMA))
                if klass == "static":
                    ana = cos_arr + ANALYTICAL_ALPHA * s
                elif klass == "motion":
                    ana = cos_arr - ANALYTICAL_ALPHA * s
                else:
                    ana = cos_arr.copy()

                # Learned gate (batched over n frames)
                raw_t = torch.tensor(cos_arr, dtype=torch.float32, device=device)
                d_t = torch.full((n,), d, dtype=torch.float32, device=device)
                emb_t = emb.expand(n, -1)
                oh_t = oh.expand(n, -1)
                with torch.no_grad():
                    lrn_t = gate.predict(raw_t, d_t, oh_t, emb_t)
                lrn = lrn_t.detach().cpu().numpy()

                bucket = "gt" if tid in gt_tids else "ngt"
                pooled[sentence][f"raw_{bucket}"].extend(cos_arr.tolist())
                pooled[sentence][f"ana_{bucket}"].extend(ana.tolist())
                pooled[sentence][f"lrn_{bucket}"].extend(lrn.tolist())

    # Compute per-expression seps
    per_expr = []
    for sentence, p in pooled.items():
        raw_gt = np.asarray(p["raw_gt"], dtype=np.float32)
        raw_ngt = np.asarray(p["raw_ngt"], dtype=np.float32)
        if len(raw_gt) == 0 or len(raw_ngt) == 0:
            continue
        ana_gt = np.asarray(p["ana_gt"], dtype=np.float32)
        ana_ngt = np.asarray(p["ana_ngt"], dtype=np.float32)
        lrn_gt = np.asarray(p["lrn_gt"], dtype=np.float32)
        lrn_ngt = np.asarray(p["lrn_ngt"], dtype=np.float32)
        per_expr.append({
            "sentence": sentence,
            "expr_class": p["expr_class"],
            "n_gt": int(len(raw_gt)),
            "n_ngt": int(len(raw_ngt)),
            "raw_sep": float(raw_gt.mean() - raw_ngt.mean()),
            "ana_sep": float(ana_gt.mean() - ana_ngt.mean()),
            "lrn_sep": float(lrn_gt.mean() - lrn_ngt.mean()),
            # also keep raw means for sanity / pooling
            "raw_gt_mean": float(raw_gt.mean()),
            "raw_ngt_mean": float(raw_ngt.mean()),
            "ana_gt_mean": float(ana_gt.mean()),
            "ana_ngt_mean": float(ana_ngt.mean()),
            "lrn_gt_mean": float(lrn_gt.mean()),
            "lrn_ngt_mean": float(lrn_ngt.mean()),
        })

    # Pooled (across all expressions, all frames) sep
    def pooled_sep(key_gt: str, key_ngt: str) -> float:
        gt_all, ngt_all = [], []
        for p in pooled.values():
            gt_all.extend(p[key_gt])
            ngt_all.extend(p[key_ngt])
        if not gt_all or not ngt_all:
            return 0.0
        return float(np.mean(gt_all) - np.mean(ngt_all))

    summary = {
        "raw_pooled_sep": pooled_sep("raw_gt", "raw_ngt"),
        "ana_pooled_sep": pooled_sep("ana_gt", "ana_ngt"),
        "lrn_pooled_sep": pooled_sep("lrn_gt", "lrn_ngt"),
        # Macro = mean over per-expression seps (each expression weighted equally)
        "raw_macro_sep": float(np.mean([r["raw_sep"] for r in per_expr])) if per_expr else 0.0,
        "ana_macro_sep": float(np.mean([r["ana_sep"] for r in per_expr])) if per_expr else 0.0,
        "lrn_macro_sep": float(np.mean([r["lrn_sep"] for r in per_expr])) if per_expr else 0.0,
        "n_lrn_beats_ana": int(sum(1 for r in per_expr if r["lrn_sep"] > r["ana_sep"])),
        "n_lrn_beats_raw": int(sum(1 for r in per_expr if r["lrn_sep"] > r["raw_sep"])),
        "n_exprs": len(per_expr),
    }
    return per_expr, summary


def write_report(per_expr, summary, weights, gate_path, seqs, out_md, out_json):
    # Sort by Δ vs analytical to spotlight wins/losses
    rows = sorted(per_expr, key=lambda r: r["lrn_sep"] - r["ana_sep"], reverse=True)

    # Canary checks
    def find(name_substr_set):
        out = []
        for r in per_expr:
            s = r["sentence"].lower()
            if any(sub in s for sub in name_substr_set):
                out.append(r)
        return out

    canaries = {
        "braking": find({"braking"}),
        "parking_cars_or_vehicles": find({"parking cars", "parking vehicles"}),
        "turning_cars_or_vehicles": find({"turning cars", "turning vehicles"}),
        "moving_pedestrian": find({"moving pedestrian"}),
        "cars_in_front_of_ours": find({"cars in front of ours"}),
    }

    def avg_sep(rs, key):
        if not rs:
            return None
        return float(np.mean([r[key] for r in rs]))

    canary_summary = {}
    for name, rs in canaries.items():
        canary_summary[name] = {
            "n_exprs": len(rs),
            "raw_sep": avg_sep(rs, "raw_sep"),
            "ana_sep": avg_sep(rs, "ana_sep"),
            "lrn_sep": avg_sep(rs, "lrn_sep"),
        }

    braking_lrn = canary_summary["braking"]["lrn_sep"]
    braking_ok = braking_lrn is not None and braking_lrn >= 0.35
    pooled_ok = summary["lrn_pooled_sep"] > summary["ana_pooled_sep"]
    verdict = "PASS" if (braking_ok and pooled_ok) else "FAIL"

    # JSON
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "weights": weights,
            "gate_path": gate_path,
            "seqs": seqs,
            "summary": summary,
            "canaries": canary_summary,
            "verdict": verdict,
            "per_expr": per_expr,
        }, f, indent=2)

    # Markdown
    lines = []
    lines.append("# Learned posthoc state-aware gate — V1 holdout sep")
    lines.append("")
    lines.append(f"Stage1 weights: `{weights}` (frozen)")
    lines.append(f"Gate weights: `{gate_path}`")
    lines.append(f"Holdout seqs: {', '.join(seqs)}")
    lines.append(f"Analytical baseline: alpha={ANALYTICAL_ALPHA}, sigma={ANALYTICAL_SIGMA}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("Pooled = single mean across every (track, frame, expr) row.")
    lines.append("Macro  = mean of per-expression seps (each expression weighted equally).")
    lines.append("")
    lines.append("| metric | raw | analytical | learned | Δ lrn vs raw | Δ lrn vs ana |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| pooled sep | {summary['raw_pooled_sep']:+.4f} | {summary['ana_pooled_sep']:+.4f} | "
        f"{summary['lrn_pooled_sep']:+.4f} | {summary['lrn_pooled_sep']-summary['raw_pooled_sep']:+.4f} | "
        f"{summary['lrn_pooled_sep']-summary['ana_pooled_sep']:+.4f} |"
    )
    lines.append(
        f"| macro sep  | {summary['raw_macro_sep']:+.4f} | {summary['ana_macro_sep']:+.4f} | "
        f"{summary['lrn_macro_sep']:+.4f} | {summary['lrn_macro_sep']-summary['raw_macro_sep']:+.4f} | "
        f"{summary['lrn_macro_sep']-summary['ana_macro_sep']:+.4f} |"
    )
    lines.append("")
    lines.append(
        f"Per-expression win counts (out of {summary['n_exprs']}): "
        f"learned > analytical in **{summary['n_lrn_beats_ana']}**; "
        f"learned > raw in **{summary['n_lrn_beats_raw']}**."
    )
    lines.append("")
    lines.append(f"**Verdict (spec criterion = pooled sep): {verdict}**")
    lines.append("")
    lines.append("Pass criterion (spec): learned pooled sep > analytical pooled sep AND braking sep >= +0.350.")
    lines.append("")
    lines.append("## Canary checks")
    lines.append("")
    lines.append("| canary | n_exprs | raw_sep | analytical_sep | learned_sep | target |")
    lines.append("|---|---|---|---|---|---|")
    targets = {
        "braking": ">= +0.350",
        "parking_cars_or_vehicles": ">= +0.100",
        "turning_cars_or_vehicles": ">= +0.200",
        "moving_pedestrian": ">= +0.050",
        "cars_in_front_of_ours": ">= +0.400",
    }
    for name, c in canary_summary.items():
        def fmt(v):
            return f"{v:+.3f}" if v is not None else "—"
        lines.append(f"| {name} | {c['n_exprs']} | {fmt(c['raw_sep'])} | {fmt(c['ana_sep'])} | {fmt(c['lrn_sep'])} | {targets.get(name,'')} |")
    lines.append("")
    lines.append("## Per-expression (sorted by Δ learned vs analytical)")
    lines.append("")
    lines.append("| Expression | class | n_gt | n_ngt | raw_sep | analytical_sep | learned_sep | Δ vs analytical | Δ vs raw |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['sentence']} | {r['expr_class']} | {r['n_gt']:,} | {r['n_ngt']:,} | "
            f"{r['raw_sep']:+.3f} | {r['ana_sep']:+.3f} | {r['lrn_sep']:+.3f} | "
            f"{r['lrn_sep']-r['ana_sep']:+.3f} | {r['lrn_sep']-r['raw_sep']:+.3f} |"
        )
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md} and {out_json}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--gate", default=DEFAULT_GATE)
    ap.add_argument("--seqs", nargs="+", default=HOLDOUT_SEQS)
    ap.add_argument("--out-md", default="diagnostics/results/multiseq/learned_gate_results.md")
    ap.add_argument("--out-json", default="diagnostics/results/multiseq/learned_gate_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")
    print(f"Gate: {args.gate}")
    print(f"Seqs: {args.seqs}")

    per_expr, summary = evaluate_all(args.weights, args.gate, args.seqs, device)
    print("\n=== Summary ===")
    print(f"  raw  pooled sep: {summary['raw_pooled_sep']:+.4f}    macro: {summary['raw_macro_sep']:+.4f}")
    print(f"  ana  pooled sep: {summary['ana_pooled_sep']:+.4f}    macro: {summary['ana_macro_sep']:+.4f}")
    print(f"  lrn  pooled sep: {summary['lrn_pooled_sep']:+.4f}    macro: {summary['lrn_macro_sep']:+.4f}")
    print(f"  per-expr wins lrn>ana: {summary['n_lrn_beats_ana']}/{summary['n_exprs']}, lrn>raw: {summary['n_lrn_beats_raw']}/{summary['n_exprs']}")
    verdict = write_report(per_expr, summary, args.weights, args.gate, args.seqs, args.out_md, args.out_json)
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
