"""
Loss functions for the GMC-Link alignment network.
"""
import torch
from torch import nn


class AlignmentLoss(nn.Module):
    """
    Symmetric InfoNCE loss with False-Negative Masking for cross-modal alignment.

    Standard CLIP-style contrastive loss assumes only the diagonal of the NxN
    similarity matrix contains positive pairs. This breaks when multiple samples
    in a batch share the same sentence (creating false negatives on the diagonal).

    FNM fix: mask out all off-diagonal pairs that share the same sentence from
    the denominator, so they are neither positive nor negative — just ignored.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sim_matrix, sentence_ids):
        """
        Args:
            sim_matrix:   (B, B) cosine similarity matrix from model.forward()
            sentence_ids: (B,) integer tensor mapping each sample to its unique sentence

        Returns:
            Scalar loss (mean of motion→language and language→motion directions)
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        # Build false-negative mask: fn_mask[i][j] = True if same sentence AND i != j
        sid_row = sentence_ids.unsqueeze(1)  # (B, 1)
        sid_col = sentence_ids.unsqueeze(0)  # (1, B)
        same_sentence = (sid_row == sid_col)  # (B, B) bool
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        fn_mask = same_sentence & ~diag_mask  # off-diagonal same-sentence pairs

        # Motion → Language direction
        m2l_exp = torch.exp(sim_matrix)
        m2l_exp_masked = m2l_exp * (~fn_mask).float()  # zero out FN contributions
        m2l_numerator = torch.diag(m2l_exp)  # (B,)
        m2l_denominator = m2l_exp_masked.sum(dim=1)  # (B,)
        m2l_loss = -torch.log(m2l_numerator / (m2l_denominator + 1e-8))

        # Language → Motion direction (transpose)
        l2m_exp = torch.exp(sim_matrix.t())
        l2m_exp_masked = l2m_exp * (~fn_mask.t()).float()
        l2m_numerator = torch.diag(l2m_exp)
        l2m_denominator = l2m_exp_masked.sum(dim=1)
        l2m_loss = -torch.log(l2m_numerator / (l2m_denominator + 1e-8))

        return (m2l_loss.mean() + l2m_loss.mean()) / 2.0
