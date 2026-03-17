"""
Take stabilized velocity vector from utils.py, and align it
with language features from the language model using a small MLP.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MotionLanguageAligner(nn.Module):
    """
    Reasoning Head of GMC-Link that aligns motion features with language features.

    Trained with Supervised InfoNCE. At inference, use encode() to get L2-normalized
    embeddings and compute cosine similarity directly.
    """

    def __init__(
        self, motion_dim: int = 9, lang_dim: int = 768, embed_dim: int = 256
    ) -> None:
        super().__init__()
        # Motion Encoder: Project 9D vector (dx, dy, dw, dh, cx, cy, w, h, snr) into a semantic vector.
        # Wider MLP to learn nuanced motion semantics (e.g., turning vs moving forward)
        self.motion_projector = nn.Sequential(
            nn.Linear(motion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Language Projector: two-layer projection with bottleneck
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def encode(
        self, motion_feats: torch.Tensor, lang_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project motion and language inputs into the shared latent space.
        This is the core method used by both training (InfoNCE) and inference.

        Args:
            motion_feats: (N, motion_dim) motion vectors.
            lang_feats:   (N, lang_dim) or (M, lang_dim) language embeddings.

        Returns:
            motion_emb:   (N, embed_dim) L2-normalized motion embeddings.
            lang_emb:     (N, embed_dim) or (M, embed_dim) L2-normalized language embeddings.
        """
        motion_emb = F.normalize(self.motion_projector(motion_feats), p=2, dim=-1)
        lang_emb = F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)
        return motion_emb, lang_emb

    def forward(
        self, motion_feats: torch.Tensor, lang_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity scores between motion and language embeddings.

        Args:
            motion_feats: (N, 8) motion vectors.
            lang_feats:   (M, L_dim) language embeddings.

        Returns:
            scores: (N, M) cosine similarity in [-1, 1].
        """
        motion_emb, lang_emb = self.encode(motion_feats, lang_feats)
        return torch.matmul(motion_emb, lang_emb.t())
