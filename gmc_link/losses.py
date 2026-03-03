"""
Loss functions for the GMC-Link alignment network.

Uses CLIP-style symmetric cross-modal contrastive loss:
only motion↔language similarities are computed — no intra-modal
(motion↔motion or language↔language) comparisons.
"""
import torch
from torch import nn


class AlignmentLoss(nn.Module):
    """
    CLIP-style symmetric cross-modal contrastive loss.

    Computes motion→language and language→motion InfoNCE losses over the
    cross-modal similarity matrix.  When labels are provided, all pairs
    sharing the same expression ID are treated as positives (supervised).

    This avoids the pitfall of stacking both modalities into one feature
    vector: identical language embeddings would create trivial lang↔lang
    positives that dominate the gradient and starve the motion encoder.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_emb: torch.Tensor, lang_emb: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_emb: (N, D) L2-normalized motion embeddings.
            lang_emb:   (N, D) L2-normalized language embeddings.
            labels:     (N,) integer expression IDs.

        Returns:
            Scalar symmetric cross-modal InfoNCE loss.
        """
        # Cross-modal similarity matrix  (N, N)
        logits = torch.matmul(motion_emb, lang_emb.t()) / self.temperature

        # Positive mask: position (i, j) is 1 when labels[i] == labels[j]
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # ── motion → language ──
        log_prob_m2l = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        m2l_loss = -(pos_mask * log_prob_m2l).sum(1) / pos_mask.sum(1).clamp(min=1)

        # ── language → motion ──
        log_prob_l2m = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        l2m_loss = -(pos_mask * log_prob_l2m).sum(0) / pos_mask.sum(0).clamp(min=1)

        return (m2l_loss.mean() + l2m_loss.mean()) / 2
