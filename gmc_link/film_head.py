"""FiLM head: 13D ego-compensated motion vector → per-channel (γ, β) modulation.

Identity-initialized: γ=1, β=0 at step 0 → forward(motion, feat) returns feat unchanged.
This makes a checkpoint loaded with strict=False bit-exact to the frozen iKUN baseline
before any training, so we can verify integration without retraining.
"""
import torch
import torch.nn as nn


class MotionFiLMHead(nn.Module):
    def __init__(self, motion_dim: int = 13, hidden: int = 128, feat_dim: int = 2048):
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(motion_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),
        )
        # Identity init on final layer only — γ = 1 + 0, β = 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, motion_13d: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        motion_13d: [BT, 13]  per-sample 13D vector
        feat:       [HW, BT, C]  visual fusion feature (e.g. [49, BT, 2048])
        returns:    [HW, BT, C]
        """
        gb = self.mlp(motion_13d)              # [BT, 2C]
        gamma_delta, beta = gb.chunk(2, dim=-1)  # each [BT, C]
        gamma = 1.0 + gamma_delta              # identity at zero-weighted final layer
        gamma = gamma.unsqueeze(0)             # [1, BT, C]
        beta = beta.unsqueeze(0)               # [1, BT, C]
        return gamma * feat + beta
