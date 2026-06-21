"""
Loss functions for the GMC-Link alignment network.
"""
import torch
from torch import nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    Symmetric InfoNCE loss with fixed temperature.

    Standard CLIP-style cross-modal contrastive loss:
      L = CE(sim / τ, targets)  where targets = [0, 1, ..., B-1] (diagonal)
    Applied symmetrically: motion→language + language→motion.
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = False):
        super().__init__()
        if learnable:
            # Store as log(1/τ) so exp gives 1/τ, keeping τ positive
            import math
            self.log_inv_temp = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        else:
            self.log_inv_temp = None
        self._init_temperature = temperature

    @property
    def temperature(self):
        if self.log_inv_temp is not None:
            return 1.0 / self.log_inv_temp.exp().item()
        return self._init_temperature

    def forward(self, sim_matrix, sentence_ids=None, anchor_mask=None):
        """
        Args:
            sim_matrix:   (B, B) cosine similarity matrix from model.forward()
            sentence_ids: unused, kept for API compatibility
            anchor_mask:  optional (B,) bool/float tensor — only masked anchors
                          contribute to loss. Negatives stay full-batch (cross-class
                          retention). Used by per-class specialist training.

        Returns:
            Scalar loss (mean of motion→language and language→motion directions)
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        if self.log_inv_temp is not None:
            logits = sim_matrix * self.log_inv_temp.exp()  # sim * (1/τ)
        else:
            logits = sim_matrix / self._init_temperature

        # Targets: diagonal pairs are positives
        targets = torch.arange(B, device=device)

        if anchor_mask is None:
            m2l_loss = F.cross_entropy(logits, targets)
            l2m_loss = F.cross_entropy(logits.t(), targets)
            return (m2l_loss + l2m_loss) / 2.0

        # Weighted CE: only target-class anchors contribute
        w = anchor_mask.to(dtype=logits.dtype, device=device)
        norm = w.sum().clamp_min(1.0)
        m2l_per = F.cross_entropy(logits, targets, reduction="none")
        l2m_per = F.cross_entropy(logits.t(), targets, reduction="none")
        m2l_loss = (m2l_per * w).sum() / norm
        l2m_loss = (l2m_per * w).sum() / norm
        return (m2l_loss + l2m_loss) / 2.0


class StructuralConsensusLoss(nn.Module):
    """Pairwise-distance (and optional triplet-angle) consensus between motion
    and language embedding manifolds.

    For L2-normalized embeddings z_m, z_l in (B, D):
        L_dist  = MSE( D(z_m)/mean(D(z_m)), D(z_l)/mean(D(z_l)) )
    where D = pairwise euclidean. Scale-normalized so the loss is invariant to
    average inter-sample spread; only the *relative* geometry must match.

    Triplet-angle term (optional, mode="dist_angle"):
        For random anchor i with two neighbours j, k, match
        cos((z_j - z_i),(z_k - z_i)) across modalities.
    """

    def __init__(self, lam_angle: float = 0.5, n_triplets: int = 1024,
                 mode: str = "dist"):
        super().__init__()
        assert mode in ("dist", "dist_angle"), f"unknown mode={mode}"
        self.lam_angle = lam_angle
        self.n_triplets = n_triplets
        self.mode = mode

    @staticmethod
    def _pairwise_mse(z_m, z_l):
        D_m = torch.cdist(z_m, z_m, p=2)
        D_l = torch.cdist(z_l, z_l, p=2)
        D_m = D_m / D_m.mean().clamp_min(1e-8)
        D_l = D_l / D_l.mean().clamp_min(1e-8)
        return F.mse_loss(D_m, D_l)

    def _triplet_angle_mse(self, z_m, z_l):
        B = z_m.size(0)
        if B < 3:
            return z_m.new_zeros(())
        device = z_m.device
        idx = torch.randint(0, B, (self.n_triplets, 3), device=device)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        same = (i == j) | (i == k) | (j == k)
        keep = ~same
        if keep.sum() == 0:
            return z_m.new_zeros(())
        i, j, k = i[keep], j[keep], k[keep]
        v_mj = F.normalize(z_m[j] - z_m[i], dim=-1, eps=1e-8)
        v_mk = F.normalize(z_m[k] - z_m[i], dim=-1, eps=1e-8)
        v_lj = F.normalize(z_l[j] - z_l[i], dim=-1, eps=1e-8)
        v_lk = F.normalize(z_l[k] - z_l[i], dim=-1, eps=1e-8)
        cos_m = (v_mj * v_mk).sum(-1)
        cos_l = (v_lj * v_lk).sum(-1)
        return F.mse_loss(cos_m, cos_l)

    def forward(self, z_m, z_l):
        """Args:
            z_m, z_l: (B, D) L2-normalized cross-modal embeddings.
        Returns scalar loss.
        """
        L = self._pairwise_mse(z_m, z_l)
        if self.mode == "dist_angle":
            L = L + self.lam_angle * self._triplet_angle_mse(z_m, z_l)
        return L


class HardNegativeInfoNCE(nn.Module):
    """Hard-negative-mining InfoNCE with optional False-Negative Masking.

    At β=0 and fnm=False, this reduces to standard InfoNCE.
    At β>0, negatives are reweighted by exp(β * sim) (Robinson et al. 2021 style).
    When fnm=True, same-sentence pairs are excluded from the negative set.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        beta: float = 1.0,
        fnm: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.fnm = fnm

    def compute_negative_weights(self, sim_matrix, sentence_ids):
        """Return (w, n_neg) for inspection.

        w: (B, B) tensor of normalized negative weights (positives are 0).
        n_neg: (B,) long tensor of negative counts per anchor.

        Invariant: w.sum(dim=1) == n_neg (with 0 == 0 for fully-masked rows).
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device
        diag = torch.eye(B, dtype=torch.bool, device=device)
        if self.fnm:
            same_sentence = sentence_ids[:, None] == sentence_ids[None, :]
        else:
            same_sentence = diag
        negative_mask = (~same_sentence) & ~diag

        log_w_raw = (self.beta * sim_matrix).masked_fill(~negative_mask, float("-inf"))
        log_w_norm = log_w_raw - torch.logsumexp(log_w_raw, dim=1, keepdim=True)
        n_neg = negative_mask.sum(dim=1).to(sim_matrix.dtype)
        log_w = log_w_norm + torch.log(n_neg.clamp_min(1)).unsqueeze(1)
        # Guard against NaN on fully-masked rows (log_w_norm was -inf − −inf = NaN there).
        log_w = log_w.masked_fill(~negative_mask, float("-inf"))
        w = log_w.exp().masked_fill(~negative_mask, 0.0)
        return w, n_neg.long()

    def forward(self, sim_matrix, sentence_ids=None, anchor_mask=None):
        if self.fnm and sentence_ids is None:
            raise ValueError("sentence_ids is required when fnm=True")
        B = sim_matrix.size(0)
        device = sim_matrix.device
        logits = sim_matrix / self.temperature

        diag = torch.eye(B, dtype=torch.bool, device=device)
        if self.fnm:
            same_sentence = sentence_ids[:, None] == sentence_ids[None, :]
        else:
            same_sentence = diag
        positive_mask = same_sentence
        negative_mask = (~positive_mask) & ~diag

        pos_logits = logits.diagonal()  # (B,)

        # ── β-weighted denominator for motion→language direction ──
        # w_raw[i,j] = exp(β * sim[i,j]) on negatives, 0 elsewhere.
        # Then normalize: w[i,:] = w_raw[i,:] / w_raw[i,:].sum() * N_neg[i]
        # so Σⱼ w[i,j] = N_neg[i] (preserves β=0 → uniform weights = 1).
        def weighted_neg_lse(lg, nm, sim_for_weights):
            log_w_raw = (self.beta * sim_for_weights).masked_fill(~nm, float("-inf"))
            log_w_norm = log_w_raw - torch.logsumexp(log_w_raw, dim=1, keepdim=True)
            n_neg = nm.sum(dim=1, keepdim=True).clamp_min(1).to(lg.dtype)
            log_w = log_w_norm + torch.log(n_neg)  # rescale so Σw = N_neg
            # Fully-masked rows produce NaN from (-inf) - (-inf); clear them so
            # they contribute -inf to the logsumexp below (i.e., no negatives).
            log_w = log_w.masked_fill(~nm, float("-inf"))

            # Weighted logsumexp: logsumexp_j( log(w[i,j]) + logits[i,j] ) over negatives
            masked_logits = lg.masked_fill(~nm, float("-inf"))
            return torch.logsumexp(log_w + masked_logits, dim=1)

        neg_lse_m2l = weighted_neg_lse(logits,     negative_mask,     sim_matrix)
        neg_lse_l2m = weighted_neg_lse(logits.t(), negative_mask.t(), sim_matrix.t())

        den_m2l = torch.logsumexp(torch.stack([pos_logits, neg_lse_m2l], dim=1), dim=1)
        den_l2m = torch.logsumexp(torch.stack([pos_logits, neg_lse_l2m], dim=1), dim=1)

        per_m2l = den_m2l - pos_logits  # (B,)
        per_l2m = den_l2m - pos_logits  # (B,)

        if anchor_mask is None:
            return (per_m2l.mean() + per_l2m.mean()) / 2.0

        w = anchor_mask.to(dtype=per_m2l.dtype, device=device)
        norm = w.sum().clamp_min(1.0)
        l_m2l = (per_m2l * w).sum() / norm
        l_l2m = (per_l2m * w).sum() / norm
        return (l_m2l + l_l2m) / 2.0
