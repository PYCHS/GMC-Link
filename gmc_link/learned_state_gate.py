"""Learned post-hoc state-aware gate.

Small MLP that adjusts a raw cosine score using:
    - raw_cos                                  (1D)
    - d_track  (long-scale ego-comp speed)     (1D)
    - s_track = exp(-d_track / sigma_default)  (1D, analytical signal)
    - expr_class one-hot (static/motion/app)   (3D)
    - expr_embedding (sentence embedding)      (lang_dim, default 384)

The output of the network is a bounded delta `Delta = 0.5 * tanh(net_out)` in
[-0.5, 0.5]; the final gated cosine is `gated = raw_cos + Delta`. The
magnitude bound mirrors the analytical gate's `alpha = 0.5` ceiling.

This module touches NO existing GMC-Link components. Stage1 aligner stays
frozen; this is purely a posthoc learned rescore on top of cosines.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# Order must match analytical gate's `classify_expr` output set.
EXPR_CLASSES = ("static", "motion", "appearance")


def expr_class_to_onehot(klass: str) -> torch.Tensor:
    """Map class name -> 3D one-hot tensor (CPU, float32)."""
    out = torch.zeros(3, dtype=torch.float32)
    if klass in EXPR_CLASSES:
        out[EXPR_CLASSES.index(klass)] = 1.0
    return out


class LearnedStateGate(nn.Module):
    """Predict a bounded cosine adjustment from posthoc features.

    Args:
        lang_dim: dimension of expression embedding (default 384, MiniLM-L6).
        hidden: hidden widths of the MLP.
        delta_bound: magnitude cap on the predicted adjustment. Default 0.5
            mirrors the analytical-gate `alpha=0.5` budget.
        sigma_default: sigma used to precompute `s_track` from `d_track` so
            the gate has the analytical signal as an explicit input. We use
            the MVP best (sigma=4.0) so the analytical recipe lives inside
            the input space — the network only has to learn corrections.
    """

    def __init__(
        self,
        lang_dim: int = 384,
        hidden: tuple[int, int] = (128, 64),
        delta_bound: float = 0.5,
        sigma_default: float = 4.0,
    ) -> None:
        super().__init__()
        self.lang_dim = lang_dim
        self.delta_bound = float(delta_bound)
        self.sigma_default = float(sigma_default)

        in_dim = 1 + 1 + 1 + 3 + lang_dim  # raw_cos + d + s + onehot + lang
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden[1], 1),
        )

    @staticmethod
    def build_features(
        raw_cos: torch.Tensor,
        d_track: torch.Tensor,
        expr_onehot: torch.Tensor,
        expr_emb: torch.Tensor,
        sigma_default: float,
    ) -> torch.Tensor:
        """Stack the input feature vector.

        All inputs (B, ...). raw_cos and d_track are (B,) or (B, 1).
        expr_onehot is (B, 3). expr_emb is (B, lang_dim).
        """
        if raw_cos.dim() == 1:
            raw_cos = raw_cos.unsqueeze(1)
        if d_track.dim() == 1:
            d_track = d_track.unsqueeze(1)
        s_track = torch.exp(-d_track / sigma_default)
        return torch.cat([raw_cos, d_track, s_track, expr_onehot, expr_emb], dim=-1)

    def forward(
        self,
        raw_cos: torch.Tensor,
        d_track: torch.Tensor,
        expr_onehot: torch.Tensor,
        expr_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return (gated_cos, delta).

        gated_cos = raw_cos + delta_bound * tanh(net(features))
        delta is the bounded adjustment used at gate time.
        """
        feats = self.build_features(
            raw_cos, d_track, expr_onehot, expr_emb, self.sigma_default
        )
        raw = self.net(feats).squeeze(-1)
        delta = self.delta_bound * torch.tanh(raw)
        if raw_cos.dim() == 2 and raw_cos.size(-1) == 1:
            base = raw_cos.squeeze(-1)
        else:
            base = raw_cos
        return base + delta, delta

    @torch.no_grad()
    def predict(
        self,
        raw_cos: torch.Tensor,
        d_track: torch.Tensor,
        expr_onehot: torch.Tensor,
        expr_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Inference helper that returns the gated cosine only."""
        gated, _ = self.forward(raw_cos, d_track, expr_onehot, expr_emb)
        return gated
