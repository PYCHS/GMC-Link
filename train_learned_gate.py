#!/usr/bin/env python3
"""Collect features and train the learned posthoc state-aware gate.

Reuses heavy machinery from `run_posthoc_state_gate_mvp.py` (per-track
stationarity, raw cosine collection from the frozen stage1 aligner, expr
classification). Saves a tensor cache to avoid recomputation between epochs
and across train/eval scripts.

Output:
    cache/learned_gate_features_v1train.pt   — per-(seq, expr, tid) tuples
    learned_gate_v1train.pt                  — trained gate weights

Run:
    ~/miniconda/envs/RMOT/bin/python train_learned_gate.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gmc_link.learned_state_gate import LearnedStateGate, EXPR_CLASSES, expr_class_to_onehot
from gmc_link.text_utils import TextEncoder
from run_posthoc_state_gate_mvp import (
    collect_raw_cosines_for_seq,
    classify_expr,
)


V1_TRAIN_SEQS = [
    "0001", "0002", "0003", "0004", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0016", "0017", "0018", "0019", "0020",
]

DEFAULT_WEIGHTS = "gmc_link_weights_v1train_stage1.pth"
CACHE_PATH = "cache/learned_gate_features_v1train.pt"
GATE_OUT = "learned_gate_v1train.pt"


# ---- Feature collection -----------------------------------------------------

def collect_features(
    weights: str,
    seqs: List[str],
    device: torch.device,
    cache_path: str,
    overwrite: bool = False,
):
    """For each seq run collect_raw_cosines_for_seq, then explode into a flat
    list of (raw_cos, d_track, expr_class, expr_idx, label, sentence, seq, tid)
    rows. Each "row" is a single (track, expression, frame) cosine — the gate
    operates per-frame because that matches sep computation.

    A separate dict maps sentence -> 384D embedding so we don't blow the cache
    up by repeating the embedding 3M times.
    """
    if os.path.exists(cache_path) and not overwrite:
        print(f"[cache] reusing {cache_path}")
        return torch.load(cache_path, weights_only=False)

    encoder = TextEncoder(model_name="all-MiniLM-L6-v2", device=str(device))

    raw_cos_all: List[float] = []
    d_track_all: List[float] = []
    expr_idx_all: List[int] = []   # index into expr_table
    label_all: List[float] = []
    seq_id_all: List[str] = []
    tid_all: List[int] = []
    expr_table: List[dict] = []    # [{sentence, expr_class, embedding}]
    sent_to_idx: Dict[str, int] = {}

    for seq in seqs:
        print(f"\n=== Collecting seq {seq} ===")
        expr_dir = os.path.join("refer-kitti", "expression", seq)
        if not os.path.isdir(expr_dir):
            print(f"[skip] seq={seq}: no expression dir at {expr_dir}")
            continue
        seq_data = collect_raw_cosines_for_seq(weights, seq, device)
        if seq_data is None:
            print(f"[skip] seq={seq} returned None")
            continue
        d_track = seq_data["d_track"]
        for expr_rec in seq_data["expressions"]:
            sentence = expr_rec["sentence"]
            klass = expr_rec["expr_class"]
            gt_tids = expr_rec["gt_tids"]

            if sentence not in sent_to_idx:
                emb = encoder.encode(sentence).detach().cpu().squeeze(0).float()
                sent_to_idx[sentence] = len(expr_table)
                expr_table.append({
                    "sentence": sentence,
                    "expr_class": klass,
                    "embedding": emb,
                })
            expr_idx = sent_to_idx[sentence]

            for tid, cos_arr in expr_rec["track_cos"].items():
                if tid not in d_track:
                    continue
                d = float(d_track[tid])
                lbl = 1.0 if tid in gt_tids else 0.0
                # Each frame is one row.
                for c in cos_arr:
                    raw_cos_all.append(float(c))
                    d_track_all.append(d)
                    expr_idx_all.append(expr_idx)
                    label_all.append(lbl)
                    seq_id_all.append(seq)
                    tid_all.append(int(tid))

    cache = {
        "raw_cos": torch.tensor(raw_cos_all, dtype=torch.float32),
        "d_track": torch.tensor(d_track_all, dtype=torch.float32),
        "expr_idx": torch.tensor(expr_idx_all, dtype=torch.long),
        "label": torch.tensor(label_all, dtype=torch.float32),
        "seq_id": seq_id_all,        # list[str]
        "tid": torch.tensor(tid_all, dtype=torch.long),
        "expr_table": expr_table,    # list[{sentence, expr_class, embedding}]
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(cache, cache_path)
    print(f"\n[cache] saved {cache_path}: n={len(raw_cos_all):,}, exprs={len(expr_table)}")
    return cache


# ---- Training ---------------------------------------------------------------

def margin_loss_train(
    cache: dict,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    wd: float = 1e-4,
    margin: float = 0.3,
    pairs_per_expr: int = 32,
    seed: int = 0,
) -> LearnedStateGate:
    """Train the gate with a pairwise margin loss.

    Each minibatch:
        for each expression with both pos and neg rows, sample
        `pairs_per_expr` (pos, neg) frames; the loss is
        max(0, margin - (gated_pos - gated_neg)).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_cos = cache["raw_cos"]
    d_track = cache["d_track"]
    expr_idx = cache["expr_idx"]
    label = cache["label"]
    expr_table = cache["expr_table"]
    n_expr = len(expr_table)

    # Precompute per-expr positive / negative indices
    pos_idx_by_expr: Dict[int, np.ndarray] = {}
    neg_idx_by_expr: Dict[int, np.ndarray] = {}
    expr_idx_np = expr_idx.numpy()
    label_np = label.numpy()
    for e in range(n_expr):
        rows = np.where(expr_idx_np == e)[0]
        pos = rows[label_np[rows] == 1.0]
        neg = rows[label_np[rows] == 0.0]
        if len(pos) > 0 and len(neg) > 0:
            pos_idx_by_expr[e] = pos
            neg_idx_by_expr[e] = neg
    trainable_exprs = sorted(pos_idx_by_expr.keys())
    print(f"[train] {len(trainable_exprs)}/{n_expr} expressions usable (need both pos+neg)")

    # Stack expr embeddings + onehots into tensors keyed by expr_idx
    expr_emb_table = torch.stack([e["embedding"] for e in expr_table]).to(device)        # (E, 384)
    expr_oh_table = torch.stack([
        expr_class_to_onehot(e["expr_class"]) for e in expr_table
    ]).to(device)                                                                         # (E, 3)

    raw_cos = raw_cos.to(device)
    d_track = d_track.to(device)

    gate = LearnedStateGate(lang_dim=expr_emb_table.shape[1]).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=wd)

    rng = np.random.default_rng(seed)
    log = []

    for epoch in range(epochs):
        gate.train()
        rng.shuffle(trainable_exprs)
        total = 0.0
        nb = 0

        # Build one batch per ~32 expressions to keep gradient signal stable.
        BATCH_EXPRS = 32
        for chunk_start in range(0, len(trainable_exprs), BATCH_EXPRS):
            chunk = trainable_exprs[chunk_start: chunk_start + BATCH_EXPRS]

            pos_rows, neg_rows, expr_rows = [], [], []
            for e in chunk:
                pos = pos_idx_by_expr[e]
                neg = neg_idx_by_expr[e]
                k = pairs_per_expr
                p_choice = rng.choice(pos, size=k, replace=len(pos) < k)
                n_choice = rng.choice(neg, size=k, replace=len(neg) < k)
                pos_rows.append(p_choice)
                neg_rows.append(n_choice)
                expr_rows.extend([e] * k)
            pos_idx_arr = torch.tensor(np.concatenate(pos_rows), dtype=torch.long, device=device)
            neg_idx_arr = torch.tensor(np.concatenate(neg_rows), dtype=torch.long, device=device)
            expr_arr = torch.tensor(expr_rows, dtype=torch.long, device=device)

            pos_raw = raw_cos[pos_idx_arr]
            neg_raw = raw_cos[neg_idx_arr]
            pos_d = d_track[pos_idx_arr]
            neg_d = d_track[neg_idx_arr]
            emb = expr_emb_table[expr_arr]
            oh = expr_oh_table[expr_arr]

            pos_gated, _ = gate(pos_raw, pos_d, oh, emb)
            neg_gated, _ = gate(neg_raw, neg_d, oh, emb)
            loss = F.relu(margin - (pos_gated - neg_gated)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1

        avg = total / max(nb, 1)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")
        log.append(avg)

    return gate, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--seqs", nargs="+", default=V1_TRAIN_SEQS)
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--out", default=GATE_OUT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--overwrite-cache", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Weights: {args.weights}")
    print(f"Train seqs ({len(args.seqs)}): {args.seqs}")

    cache = collect_features(args.weights, args.seqs, device, args.cache, args.overwrite_cache)
    print(f"\n[cache stats] rows={len(cache['raw_cos']):,}  exprs={len(cache['expr_table'])}")
    n_pos = float((cache["label"] == 1).sum())
    n_neg = float((cache["label"] == 0).sum())
    print(f"  pos={n_pos:,.0f}  neg={n_neg:,.0f}  pos_rate={n_pos/(n_pos+n_neg):.4f}")

    gate, log = margin_loss_train(
        cache, device,
        epochs=args.epochs, lr=args.lr,
        margin=args.margin, pairs_per_expr=args.pairs,
    )
    torch.save({
        "model": gate.state_dict(),
        "lang_dim": gate.lang_dim,
        "delta_bound": gate.delta_bound,
        "sigma_default": gate.sigma_default,
        "loss_log": log,
        "config": {
            "epochs": args.epochs, "lr": args.lr, "margin": args.margin,
            "pairs_per_expr": args.pairs, "weights": args.weights,
            "seqs": args.seqs,
        },
    }, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
