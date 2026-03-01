"""
GMC-Link Training Script with Homography Learning
==================================================
Trains the MotionLanguageAlignerWithHomography to learn camera motion compensation
from data instead of using geometric preprocessing.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Tuple, Optional, Dict
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from gmc_link.losses import AlignmentLoss
from gmc_link.alignment_with_homography import MotionLanguageAlignerWithHomography
from gmc_link.dataset_with_homography import (
    MotionLanguageWithHomographyDataset,
    collate_fn_with_homography,
    build_training_data_with_homography,
)
from gmc_link.text_utils import TextEncoder


def train_one_epoch(
    model: MotionLanguageAlignerWithHomography,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_func: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for _, (motion_features, homography_features, language_features, labels) in enumerate(
        dataloader
    ):
        motion_features = motion_features.to(device)
        homography_features = homography_features.to(device)
        language_features = language_features.to(device)
        labels = labels.to(device)

        # Per-pair similarity scores with homography
        scores = model.score_pairs(motion_features, homography_features, language_features)
        loss = loss_func(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track accuracy
        preds = (torch.sigmoid(scores) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def setup_data(
    device: torch.device,
    kitti_root: str,
    refer_kitti_root: str,
    sequences: list,
    batch_size: int,
    frame_gap: int = 5,
) -> Optional[DataLoader]:
    """
    Initialize text encoder, build training dataset with homography, and return a DataLoader.
    """
    print("Loading text encoder...")
    encoder = TextEncoder(device=str(device))

    print("Encoding all sentences...")
    # Collect all unique sentences from all sequences
    from gmc_link.dataset import load_refer_kitti_expressions, is_motion_expression
    all_sentences = set()
    for seq in sequences:
        expr_dir = os.path.join(refer_kitti_root, "expression", seq)
        expressions = load_refer_kitti_expressions(expr_dir)
        motion_expressions = [e for e in expressions if is_motion_expression(e["sentence"])]
        for expr in motion_expressions:
            all_sentences.add(expr["sentence"])
    
    sentence_embeddings: Dict[str, torch.Tensor] = {}
    for sentence in all_sentences:
        embedding = encoder.encode(sentence)
        sentence_embeddings[sentence] = embedding.cpu().numpy().squeeze()
    
    print(f"Encoded {len(sentence_embeddings)} unique sentences")

    print("Building training data with homography features...")
    motion_data, homography_data, language_data, labels = build_training_data_with_homography(
        sequences=sequences,
        kitti_root=kitti_root,
        refer_kitti_root=refer_kitti_root,
        sentence_embeddings=sentence_embeddings,
        frame_gap=frame_gap,
    )

    print(f"Total training samples: {len(motion_data)}")
    if len(motion_data) == 0:
        return None

    dataset = MotionLanguageWithHomographyDataset(
        motion_data, homography_data, language_data, labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_homography,
    )

    return dataloader


def setup_model_and_optimizer(
    device: torch.device,
    motion_dim: int,
    homography_dim: int,
    lang_dim: int,
    embed_dim: int,
    learning_rate: float,
    epochs: int,
) -> Tuple[MotionLanguageAlignerWithHomography, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Initialize model, loss, optimizer, and learning-rate scheduler.
    """
    model = MotionLanguageAlignerWithHomography(
        motion_dim=motion_dim,
        homography_dim=homography_dim,
        lang_dim=lang_dim,
        embed_dim=embed_dim,
    ).to(device)

    criterion = AlignmentLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    return model, criterion, optimizer, scheduler


def train_loop(
    model: MotionLanguageAlignerWithHomography,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    save_path: str = "gmc_link_with_homography_weights.pth",
) -> None:
    """
    Execute the main training loop across all epochs and save the final weights.
    """
    print(f"Starting training on {device} | {len(dataloader)} batches/epoch...")
    for epoch in tqdm(range(epochs)):
        avg_loss, accuracy = train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:.2%} | LR: {current_lr:.6f}"
            )

    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")


def main() -> None:
    """
    Main training execution block.
    """
    # --- Configuration ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    learning_rate = 1e-3
    batch_size = 128
    epochs = 100  # Increased for better convergence
    motion_dim = 8
    homography_dim = 8
    lang_dim = 384
    embed_dim = 256

    # Refer-KITTI data paths
    kitti_root = "refer-kitti/KITTI"
    refer_kitti_root = "refer-kitti"
    
    # Train on sequences 0015, 0016, 0018 (test on 0011)
    # Train on all sequences except 0011 (test sequence)
    sequences = [
        "0001", "0002", "0003", "0004", "0005", 
        "0006", "0007", "0008", "0009", "0010",
        "0012", "0013", "0014", "0015", "0016", "0018", "0020"
    ]

    # --- Pipeline ---
    dataloader = setup_data(
        device, kitti_root, refer_kitti_root, sequences, batch_size
    )
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, motion_dim, homography_dim, lang_dim, embed_dim, learning_rate, epochs
    )

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, epochs)


if __name__ == "__main__":
    main()
