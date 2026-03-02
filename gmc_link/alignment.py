"""
Take stabilized velocity vector from utils.py, and align it
with language features from the language model using a small MLP.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MotionLanguageAligner(nn.Module):
    """
    Reasoning Head of GMC-Link that aligns motion features with language features.
    """

    def __init__(
        self, motion_dim: int = 8, lang_dim: int = 768, embed_dim: int = 256
    ) -> None:
        super().__init__()
        # Motion Encoder: Project (dx, dy, cx, cy, w, h) into a semantic vector.
        # Deeper MLP to learn nuanced motion semantics (e.g., turning vs moving forward)
        self.motion_projector = nn.Sequential(
            nn.Linear(motion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim)
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )  # Learnable temperature parameter for scaling

    def forward(
        self, motion_feats: torch.Tensor, lang_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        The 'Thinking' phase: Link geometric motion to linguistic intent.
        Used at inference time to produce scaled similarity logits.

        Args:
            motion_feats: (N, 8) Tensor of normalized world velocities [dx, dy], depth velocities [dw, dh], and positions [cx, cy, w, h].
            lang_feats: (M, L_dim) Tensor of text features representing the prompt.

        Returns:
            alignment_logits: (N, M) Matrix of similarity scores between each motion and the language concept.
        """
        motion_emb, lang_emb = self.encode(motion_feats, lang_feats)

        # (N, embed_dim) @ (embed_dim, M) -> (N, M)
        raw_similarity = torch.matmul(motion_emb, lang_emb.t())

        # Temperature Scaling: sharpen scores so the best match is distinct
        alignment_logits = raw_similarity * self.logit_scale.exp()

        return alignment_logits

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

    def score_pairs(
        self, motion_feats: torch.Tensor, lang_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-pair similarity scores for BCE training.

        Args:
            motion_feats: (N, 8) Tensor of velocity, depth scaling, and position vectors.
            lang_feats: (N, L_dim) Tensor of language embeddings (one per motion).

        Returns:
            scores: (N,) Tensor of scalar similarity scores per pair.
        """
        motion_latents, language_latents = self.encode(motion_feats, lang_feats)

        # Element-wise dot product → per-pair cosine similarity
        scores = (motion_latents * language_latents).sum(dim=-1)
        scores = scores * self.logit_scale.exp()
        return scores
