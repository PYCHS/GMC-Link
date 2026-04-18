"""Unit tests for the multi-sequence aggregator.

Uses synthetic .npz fixtures that match the schema produced by
diagnostics/diag_gt_cosine_distributions.py (Task 2). No real inference.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_npz_dir(tmp_path: Path) -> Path:
    """Write fake per-(seq, model) .npz files under tmp_path/multiseq/.

    Three seqs, two weights. Deterministic cosines so AUC is predictable.

    Model A: GT scores > non-GT scores on all seqs/exprs (clean positive).
    Model B: GT scores < non-GT scores on seq 0005 only (seq-dependent).
    """
    out_dir = tmp_path / "multiseq"
    out_dir.mkdir()

    seqs = ["0005", "0011", "0013"]
    sentences = ["moving cars", "parking cars", "cars in left"]

    def _build(model_tag: str, invert_on: set[str], seed: int) -> None:
        rng = np.random.default_rng(seed)
        for s in seqs:
            results = []
            gt_list = []
            nongt_list = []
            for sent in sentences:
                # 20 GT, 80 non-GT per expression per seq — realistic-ish ratio
                base_gt = rng.normal(0.3, 0.1, size=20).astype(np.float32)
                base_nongt = rng.normal(0.1, 0.1, size=80).astype(np.float32)
                if s in invert_on:
                    gt, nongt = base_nongt[:20], base_gt.tolist() + rng.normal(0.3, 0.1, size=60).astype(np.float32).tolist()
                    gt = np.array(gt, dtype=np.float32)
                    nongt = np.array(nongt, dtype=np.float32)
                else:
                    gt, nongt = base_gt, base_nongt
                results.append({
                    "sentence": sent,
                    "n_gt": int(len(gt)),
                    "n_nongt": int(len(nongt)),
                    "gt_mean": float(gt.mean()),
                    "gt_std": float(gt.std()),
                    "nongt_mean": float(nongt.mean()),
                    "nongt_std": float(nongt.std()),
                    "separation": float(gt.mean() - nongt.mean()),
                    "auc": 0.0,  # aggregator recomputes from raw arrays
                })
                gt_list.append(gt)
                nongt_list.append(nongt)
            path = out_dir / f"layer3_{s}_{model_tag}.npz"
            np.savez(
                path,
                results=results,
                gt_cosines_by_expr=np.array(gt_list, dtype=object),
                nongt_cosines_by_expr=np.array(nongt_list, dtype=object),
            )

    _build("model_A", invert_on=set(), seed=42)
    _build("model_B", invert_on={"0005"}, seed=44)
    return out_dir


def test_load_per_seq_expressions(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import load_per_seq_expressions
    data = load_per_seq_expressions(synthetic_npz_dir, "model_A", ["0005", "0011", "0013"])
    # keys: expression sentence; values: dict seq -> (gt_arr, nongt_arr)
    assert set(data.keys()) == {"moving cars", "parking cars", "cars in left"}
    assert set(data["moving cars"].keys()) == {"0005", "0011", "0013"}
    gt, nongt = data["moving cars"]["0005"]
    assert gt.shape == (20,)
    assert nongt.shape == (80,)


def test_compute_per_seq_auc_high_when_gt_dominates(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, compute_per_seq_auc,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_A", ["0005", "0011", "0013"])
    for sent in data:
        for s in ["0005", "0011", "0013"]:
            gt, nongt = data[sent][s]
            auc = compute_per_seq_auc(gt, nongt)
            assert auc > 0.85, (
                f"model_A {sent}/{s}: GT dominates so AUC should be high; got {auc:.3f}"
            )


def test_compute_per_seq_auc_low_when_inverted(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, compute_per_seq_auc,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_B", ["0005", "0011", "0013"])
    for sent in data:
        gt, nongt = data[sent]["0005"]
        auc = compute_per_seq_auc(gt, nongt)
        assert auc < 0.15, f"model_B inverted on 0005; AUC should be low; got {auc:.3f}"
        # Non-inverted seqs still good
        gt2, nongt2 = data[sent]["0011"]
        auc2 = compute_per_seq_auc(gt2, nongt2)
        assert auc2 > 0.85, f"model_B non-inverted on 0011; got {auc2:.3f}"


def test_macro_aggregation_math(synthetic_npz_dir: Path):
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, aggregate_expression,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_A", ["0005", "0011", "0013"])
    agg = aggregate_expression(data["moving cars"], seqs=["0005", "0011", "0013"])
    # macro = mean of per-seq AUCs
    expected_macro = np.mean([agg["auc_per_seq"][s] for s in ["0005", "0011", "0013"]])
    np.testing.assert_allclose(agg["auc_macro_mean"], expected_macro, rtol=1e-6)
    # std is the sample std across seqs
    expected_std = np.std([agg["auc_per_seq"][s] for s in ["0005", "0011", "0013"]])
    np.testing.assert_allclose(agg["auc_macro_std"], expected_std, rtol=1e-6)


def test_micro_aggregation_pools_before_auc(synthetic_npz_dir: Path):
    """Micro AUC must come from concatenated arrays, not averaged per-seq AUCs."""
    from diagnostics.aggregate_multiseq import (
        load_per_seq_expressions, aggregate_expression, compute_per_seq_auc,
    )
    data = load_per_seq_expressions(synthetic_npz_dir, "model_B", ["0005", "0011", "0013"])
    # On model_B, "moving cars" has one inverted seq (0005) and two clean.
    per_seq = data["moving cars"]
    gt_all = np.concatenate([per_seq[s][0] for s in ["0005", "0011", "0013"]])
    nongt_all = np.concatenate([per_seq[s][1] for s in ["0005", "0011", "0013"]])
    expected_micro = compute_per_seq_auc(gt_all, nongt_all)

    agg = aggregate_expression(per_seq, seqs=["0005", "0011", "0013"])
    np.testing.assert_allclose(agg["auc_micro"], expected_micro, rtol=1e-6)
    # sanity: macro ≠ micro when distributions differ
    assert abs(agg["auc_macro_mean"] - agg["auc_micro"]) > 1e-3


def test_expression_missing_from_seq_is_skipped_in_macro(tmp_path: Path):
    """If an expression has no data in a seq, macro skips that seq; micro uses pooled."""
    from diagnostics.aggregate_multiseq import aggregate_expression
    rng = np.random.default_rng(0)
    per_seq = {
        "0005": (rng.normal(0.3, 0.1, 20).astype(np.float32),
                 rng.normal(0.1, 0.1, 80).astype(np.float32)),
        "0011": (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),
        "0013": (rng.normal(0.3, 0.1, 25).astype(np.float32),
                 rng.normal(0.1, 0.1, 75).astype(np.float32)),
    }
    agg = aggregate_expression(per_seq, seqs=["0005", "0011", "0013"])
    # 0011 has no samples — excluded from macro
    assert "0011" not in agg["auc_per_seq"] or agg["auc_per_seq"]["0011"] is None
    assert agg["gt_count_per_seq"]["0011"] == 0
    # macro computed over 0005 + 0013 only
    valid_aucs = [agg["auc_per_seq"][s] for s in ["0005", "0013"]]
    np.testing.assert_allclose(agg["auc_macro_mean"], np.mean(valid_aucs), rtol=1e-6)
