"""
GMC-Link Architecture with Homography Learning
===============================================
Network that learns camera motion compensation instead of using geometric preprocessing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def decompose_homography_features(H: np.ndarray) -> np.ndarray:
    """
    Extract full 8-parameter homography representation (8-DOF).
    
    Args:
        H: (3, 3) homography matrix
    
    Returns:
        (8,) array: [h11, h12, h13, h21, h22, h23, h31, h32]
        (h33 is implicitly 1 after normalization)
    """
    # Normalize by h33 to ensure h33 = 1
    H_norm = H / (H[2, 2] + 1e-8)
    
    # Extract 8 parameters (excluding h33 which is 1)
    features = np.array([
        H_norm[0, 0], H_norm[0, 1], H_norm[0, 2],
        H_norm[1, 0], H_norm[1, 1], H_norm[1, 2],
        H_norm[2, 0], H_norm[2, 1]
    ], dtype=np.float32)
    
    return features


class MotionLanguageAlignerWithHomography(nn.Module):
    """
    Learns to align motion with language while learning camera compensation.
    
    Architecture:
    - Motion encoder: projects 8D image-frame motion
    - Homography encoder: projects 8D full homography features (8-DOF)
    - Fusion layer: combines motion + homography
    - Language encoder: projects text embedding
    - Similarity: cosine similarity in shared space
    """
    
    def __init__(
        self,
        motion_dim: int = 8,
        homography_dim: int = 8,
        lang_dim: int = 384,
        embed_dim: int = 256
    ) -> None:
        super().__init__()
        
        # Motion encoder (image-frame coordinates)
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
        
        # Homography encoder (camera motion features)
        self.homography_projector = nn.Sequential(
            nn.Linear(homography_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Fusion layer (combine motion + homography)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 64, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim)
        )
        
        # Language encoder
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(
        self,
        motion_feats: torch.Tensor,
        homography_feats: torch.Tensor,
        lang_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment scores with learned camera compensation.
        
        Args:
            motion_feats: (N, 8) - [dx, dy, dw, dh, cx, cy, w, h] in IMAGE frame
            homography_feats: (N, 5) - [tx, ty, sx, sy, theta] camera motion
            lang_feats: (1, L_dim) - language embedding
        
        Returns:
            alignment_logits: (N, 1) - similarity scores
        """
        # Encode image-frame motion
        motion_latents = self.motion_projector(motion_feats)  # (N, 256)
        
        # Encode camera motion
        camera_latents = self.homography_projector(homography_feats)  # (N, 64)
        
        # Fuse motion + camera features
        combined = torch.cat([motion_latents, camera_latents], dim=-1)  # (N, 320)
        fused_latents = self.fusion(combined)  # (N, 256)
        
        # Encode language
        language_latents = self.lang_projector(lang_feats)  # (1, 256)
        
        # L2 normalization
        fused_latents = F.normalize(fused_latents, p=2, dim=-1)
        language_latents = F.normalize(language_latents, p=2, dim=-1)
        
        # Compute similarity
        raw_similarity = torch.matmul(fused_latents, language_latents.t())
        alignment_logits = raw_similarity * self.logit_scale.exp()
        
        return alignment_logits
    
    def score_pairs(
        self,
        motion_feats: torch.Tensor,
        homography_feats: torch.Tensor,
        lang_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-pair similarity scores for BCE training.
        
        Args:
            motion_feats: (N, 8)
            homography_feats: (N, 5)
            lang_feats: (N, L_dim) - one per motion
        
        Returns:
            scores: (N,) - scalar similarity per pair
        """
        motion_latents = self.motion_projector(motion_feats)
        camera_latents = self.homography_projector(homography_feats)
        combined = torch.cat([motion_latents, camera_latents], dim=-1)
        fused_latents = F.normalize(self.fusion(combined), p=2, dim=-1)
        
        language_latents = F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)
        
        # Element-wise dot product
        scores = (fused_latents * language_latents).sum(dim=-1)
        scores = scores * self.logit_scale.exp()
        return scores
