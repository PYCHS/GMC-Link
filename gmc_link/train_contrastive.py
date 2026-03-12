"""
GMC-Link Contrastive Training Script
=====================================
Trains MotionLanguageAligner using InfoNCE loss with False-Negative Masking.
Negatives are implicit (other samples in the batch). FNM prevents same-sentence
pairs from being treated as negatives.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Tuple
from tqdm import tqdm
import math

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from gmc_link.losses import ContrastiveAlignmentLoss
from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import (
    ContrastiveMotionLanguageDataset,
    contrastive_collate_fn,
    build_contrastive_training_data,
)
from gmc_link.text_utils import TextEncoder


def compute_embedding_separation(
    model: MotionLanguageAligner,
    motion_feats: torch.Tensor,
    lang_feats: torch.Tensor,
    sentence_ids: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Compute mean cosine similarity for matched vs unmatched pairs.

    Returns:
        (pos_sim, neg_sim, separation) where separation = pos_sim - neg_sim
    """
    with torch.no_grad():
        motion_latents = F.normalize(model.motion_projector(motion_feats), p=2, dim=-1)
        lang_latents = F.normalize(model.lang_projector(lang_feats), p=2, dim=-1)
        sim = torch.matmul(motion_latents, lang_latents.t())  # (B, B) raw cosine

        B = sim.size(0)
        sid_row = sentence_ids.unsqueeze(1)
        sid_col = sentence_ids.unsqueeze(0)
        pos_mask = (sid_row == sid_col)  # same sentence = positive
        diag = torch.eye(B, dtype=torch.bool, device=sim.device)

        # Positive: diagonal pairs (true match)
        pos_sim = sim[diag].mean().item()

        # Negative: off-diagonal, different sentence
        neg_mask = ~pos_mask & ~diag
        if neg_mask.any():
            neg_sim = sim[neg_mask].mean().item()
        else:
            neg_sim = 0.0

        return pos_sim, neg_sim, pos_sim - neg_sim


def train_one_epoch(
    model: MotionLanguageAligner,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_func: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, int]:
    """
    Train for a single epoch with contrastive loss.

    Returns:
        (avg_loss, avg_pos_sim, avg_neg_sim, avg_separation, unique_sents_per_batch)
    """
    model.train()
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_unique = 0

    for motion_feats, lang_feats, sentence_ids in dataloader:
        motion_feats = motion_feats.to(device)
        lang_feats = lang_feats.to(device)
        sentence_ids = sentence_ids.to(device)

        # NxN similarity matrix from the aligner
        sim_matrix = model(motion_feats, lang_feats)

        loss = loss_func(sim_matrix, sentence_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track embedding quality
        pos_sim, neg_sim, _ = compute_embedding_separation(
            model, motion_feats, lang_feats, sentence_ids
        )
        total_pos_sim += pos_sim
        total_neg_sim += neg_sim
        total_unique += sentence_ids.unique().size(0)

    n = len(dataloader)
    return (
        total_loss / n,
        total_pos_sim / n,
        total_neg_sim / n,
        (total_pos_sim - total_neg_sim) / n,
        total_unique // n,
    )


def setup_data(
    device: torch.device,
    data_root: str,
    sequences: list,
    batch_size: int,
    frame_gap: int = 5,
):
    """Initialize text encoder, build contrastive dataset, and return DataLoader."""
    print("Loading text encoder...")
    encoder = TextEncoder(device=str(device))

    print("Building contrastive training data...")
    all_motions, all_languages, all_sids = build_contrastive_training_data(
        data_root=data_root,
        sequences=sequences,
        text_encoder=encoder,
        frame_gap=frame_gap,
    )

    print(f"Total contrastive samples: {len(all_motions)}")
    if len(all_motions) == 0:
        return None

    dataset = ContrastiveMotionLanguageDataset(all_motions, all_languages, all_sids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        drop_last=True,  # Ensure consistent batch size for contrastive learning
    )

    return dataloader


def main():
    """Main contrastive training execution."""
    # --- Configuration ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    learning_rate = 1e-3
    batch_size = 512  # Larger batch for more in-batch negatives
    epochs = 100
    lang_dim = 384

    data_root = "refer-kitti"
    sequences = [
        "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008",
        "0009", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0020",
    ]

    # --- Pipeline ---
    dataloader = setup_data(device, data_root, sequences, batch_size)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    model = MotionLanguageAligner(motion_dim=8, lang_dim=lang_dim, embed_dim=256).to(
        device
    )
    criterion = ContrastiveAlignmentLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # Theoretical loss floor
    loss_floor = math.log(batch_size)
    print(f"Starting contrastive training on {device}")
    print(f"  Batch size: {batch_size} | ln(B) floor: {loss_floor:.2f}")
    print(f"  {len(dataloader)} batches/epoch | {epochs} epochs")

    save_path = "gmc_link_contrastive_weights.pth"

    for epoch in tqdm(range(epochs)):
        avg_loss, pos_sim, neg_sim, separation, uniq = train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"\nEpoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} "
                f"(floor: {loss_floor:.2f}) | "
                f"Pos: {pos_sim:.4f} | Neg: {neg_sim:.4f} | "
                f"Sep: {separation:.4f} | Uniq/batch: {uniq} | LR: {lr:.6f}"
            )

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Weights saved to {save_path}")


if __name__ == "__main__":
    main()
