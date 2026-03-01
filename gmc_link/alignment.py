"""
Align sequential motion features with language features.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MotionLanguageAligner(nn.Module):
    """
    Reasoning head that aligns motion sequences with language features.
    """

    def __init__(
        self,
        motion_dim: int = 8,
        lang_dim: int = 768,
        embed_dim: int = 256,
        seq_length: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length

        self.motion_input = nn.Linear(motion_dim, embed_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

        self.motion_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _encode_motion(self, motion_feats: torch.Tensor) -> torch.Tensor:
        if motion_feats.dim() == 2:
            motion_feats = motion_feats.unsqueeze(1)

        t = motion_feats.shape[1]
        if t > self.seq_length:
            motion_feats = motion_feats[:, -self.seq_length :, :]
            t = self.seq_length

        key_padding_mask = (motion_feats.abs().sum(dim=-1) == 0)  # (N, T)

        x = self.motion_input(motion_feats)
        x = x + self.positional_encoding[:, :t, :]

        x = self.temporal_encoder(x, src_key_padding_mask=key_padding_mask)

        attn_logits = self.temporal_attention(x).squeeze(-1)  # (N, T)
        attn_logits = attn_logits.masked_fill(key_padding_mask, -1e9)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (N, T)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (N, E)

        motion_latents = self.motion_out(pooled)
        return motion_latents

    def _encode_lang(self, lang_feats: torch.Tensor) -> torch.Tensor:
        return self.lang_projector(lang_feats)

    def forward(self, motion_feats: torch.Tensor, lang_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_feats: (N, T, 8) motion sequence tensor (T defaults to 8).
            lang_feats: (1, L_dim) text features for the prompt.

        Returns:
            alignment_logits: (N, 1) similarity scores.
        """
        motion_latents = self._encode_motion(motion_feats)
        language_latents = self._encode_lang(lang_feats)

        motion_latents = F.normalize(motion_latents, p=2, dim=-1)
        language_latents = F.normalize(language_latents, p=2, dim=-1)

        raw_similarity = torch.matmul(motion_latents, language_latents.t())
        alignment_logits = raw_similarity * self.logit_scale.exp()
        return alignment_logits

    def score_pairs(self, motion_feats: torch.Tensor, lang_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_feats: (N, T, 8) motion sequence tensor.
            lang_feats: (N, L_dim) per-pair language embeddings.

        Returns:
            scores: (N,) similarity scores.
        """
        motion_latents = self._encode_motion(motion_feats)
        language_latents = self._encode_lang(lang_feats)

        motion_latents = F.normalize(motion_latents, p=2, dim=-1)
        language_latents = F.normalize(language_latents, p=2, dim=-1)

        scores = (motion_latents * language_latents).sum(dim=-1)
        scores = scores * self.logit_scale.exp()
        return scores
