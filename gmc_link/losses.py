"""
Loss functions for the GMC-Link alignment network.
"""
import torch
from torch import nn


class AlignmentLoss(nn.Module):
    """
    Binary cross-entropy loss for motion-language alignment.

    For each (motion, language) pair, the model predicts a scalar similarity
    score. Positive pairs (correct match) should score high, negative pairs
    (wrong sentence) should score low.

    This replaces the CLIP-style contrastive loss which breaks down when
    many samples share the same sentence.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, scores, labels):
        """
        Args:
            scores: (N,) similarity scores from the aligner
            labels: (N,) binary labels (1.0 = positive match, 0.0 = negative)
        """
        return self.loss_fn(scores, labels)


class ContrastiveAlignmentLoss(nn.Module):
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
            sim_matrix: (B, B) temperature-scaled cosine similarity matrix
                        from MotionLanguageAligner.forward()
            sentence_ids: (B,) integer tensor mapping each sample to its unique sentence

        Returns:
            Scalar loss (mean of motion→language and language→motion directions)
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        # Build false-negative mask: fn_mask[i][j] = 1 if same sentence AND i != j
        sid_row = sentence_ids.unsqueeze(1)  # (B, 1)
        sid_col = sentence_ids.unsqueeze(0)  # (1, B)
        same_sentence = (sid_row == sid_col)  # (B, B) bool
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        fn_mask = same_sentence & ~diag_mask  # off-diagonal same-sentence pairs

        # Motion → Language direction
        # For each motion_i, the positive is language_i (diagonal)
        # Denominator: all columns except false negatives
        m2l_logits = sim_matrix  # (B, B)
        m2l_exp = torch.exp(m2l_logits)
        m2l_exp_masked = m2l_exp * (~fn_mask).float()  # zero out FN contributions
        m2l_numerator = torch.diag(m2l_exp)  # (B,)
        m2l_denominator = m2l_exp_masked.sum(dim=1)  # (B,)
        m2l_loss = -torch.log(m2l_numerator / (m2l_denominator + 1e-8))

        # Language → Motion direction (transpose)
        l2m_logits = sim_matrix.t()  # (B, B)
        l2m_exp = torch.exp(l2m_logits)
        l2m_exp_masked = l2m_exp * (~fn_mask.t()).float()
        l2m_numerator = torch.diag(l2m_exp)
        l2m_denominator = l2m_exp_masked.sum(dim=1)
        l2m_loss = -torch.log(l2m_numerator / (l2m_denominator + 1e-8))

        loss = (m2l_loss.mean() + l2m_loss.mean()) / 2.0
        return loss
