"""
Evaluation Script for GMC-Link with Homography Learning
=========================================================
Evaluates the trained MotionLanguageAlignerWithHomography on sequence 0011.
Computes GT avg score and Non-GT avg score to compare with Exp 17/20 baseline.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from typing import Dict, List, Tuple

from gmc_link.alignment_with_homography import MotionLanguageAlignerWithHomography
from gmc_link.text_utils import TextEncoder
from gmc_link.dataset import load_refer_kitti_expressions, load_labels_with_ids, is_motion_expression, _extract_target_centroids
from gmc_link.core import ORBHomographyEngine
from gmc_link.dataset_with_homography import decompose_homography_features

VELOCITY_SCALE = 100.0


def evaluate_sequence(
    model: MotionLanguageAlignerWithHomography,
    encoder: TextEncoder,
    seq: str,
    kitti_root: str,
    refer_kitti_root: str,
    frame_gap: int = 5,
    device: torch.device = torch.device("cpu")
) -> Tuple[float, float, int, int]:
    """
    Evaluate model on a single sequence.
    
    Returns:
        gt_avg_score: Average alignment score for GT (correct sentence) pairs
        non_gt_avg_score: Average alignment score for Non-GT (wrong sentence) pairs
        num_gt: Number of GT pairs evaluated
        num_non_gt: Number of Non-GT pairs evaluated
    """
    print(f"\nEvaluating sequence {seq}...")
    
    # Load data
    expr_dir = os.path.join(refer_kitti_root, "expression", seq)
    labels_dir = os.path.join(refer_kitti_root, "KITTI", "labels_with_ids", "image_02", seq)
    frame_dir = os.path.join(kitti_root, "training", "image_02", seq)
    
    expressions = load_refer_kitti_expressions(expr_dir)
    motion_expressions = [e for e in expressions if is_motion_expression(e["sentence"])]
    
    print(f"  Found {len(motion_expressions)} motion expressions")
    
    # Initialize ORB engine
    orb_engine = ORBHomographyEngine(max_features=1500)
    
    frame_shape = (375, 1242)  # KITTI image size
    h, w = frame_shape
    
    gt_scores = []
    non_gt_scores = []
    
    for expr in motion_expressions:
        sentence = expr["sentence"]
        label_map = expr["label"]
        
        # Encode sentence
        lang_embedding = encoder.encode(sentence).cpu().numpy().squeeze()
        
        # Extract target centroids
        track_centroids = _extract_target_centroids(
            refer_kitti_root, seq, label_map, frame_shape=frame_shape
        )
        
        # For each track, compute motion vectors and scores
        for tid, centroids in track_centroids.items():
            sorted_frames = sorted(centroids.keys())
            for i in range(len(sorted_frames)):
                curr_fid = sorted_frames[i]
                target_fid = curr_fid + frame_gap
                
                best_j = None
                for j in range(i + 1, len(sorted_frames)):
                    if sorted_frames[j] >= target_fid:
                        best_j = j
                        break
                
                if best_j is None:
                    continue
                
                future_fid = sorted_frames[best_j]
                if future_fid - curr_fid > frame_gap * 2:
                    continue
                
                # Extract IMAGE-FRAME motion (no warping)
                cx1, cy1, bw1, bh1 = centroids[curr_fid]
                cx2, cy2, bw2, bh2 = centroids[future_fid]
                
                # IMAGE-FRAME velocity (raw pixel displacement)
                dx = (cx2 - cx1) / w * VELOCITY_SCALE
                dy = (cy2 - cy1) / h * VELOCITY_SCALE
                dw = (bw2 - bw1) / w * VELOCITY_SCALE
                dh = (bh2 - bh1) / h * VELOCITY_SCALE
                
                cx_n, cy_n = cx1 / w, cy1 / h
                bw_n, bh_n = bw1 / w, bh1 / h
                
                motion_vec = np.array([dx, dy, dw, dh, cx_n, cy_n, bw_n, bh_n], dtype=np.float32)
                
                # Compute homography
                import cv2
                curr_img_path = os.path.join(frame_dir, f"{curr_fid:06d}.png")
                future_img_path = os.path.join(frame_dir, f"{future_fid:06d}.png")
                
                img_curr = cv2.imread(curr_img_path)
                img_future = cv2.imread(future_img_path)
                
                if img_curr is not None and img_future is not None:
                    homography = orb_engine.estimate_homography(
                        img_curr, img_future, prev_bboxes=None
                    )
                else:
                    homography = None
                
                if homography is not None:
                    h_features = decompose_homography_features(homography)
                    h_features[0] /= w  # Normalize tx
                    h_features[1] /= h  # Normalize ty
                else:
                    h_features = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
                
                # Convert to tensors
                motion_tensor = torch.tensor(motion_vec, dtype=torch.float32).unsqueeze(0).to(device)
                homography_tensor = torch.tensor(h_features, dtype=torch.float32).unsqueeze(0).to(device)
                lang_tensor = torch.tensor(lang_embedding, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Compute score
                with torch.no_grad():
                    score = model.score_pairs(motion_tensor, homography_tensor, lang_tensor)
                    score = torch.sigmoid(score).item()
                
                # This is a GT pair (correct sentence for this track)
                gt_scores.append(score)
                
                # Also test with a wrong sentence (Non-GT)
                # Pick a different motion expression
                other_expressions = [e for e in motion_expressions if e["sentence"] != sentence]
                if other_expressions:
                    wrong_sentence = other_expressions[0]["sentence"]
                    wrong_embedding = encoder.encode(wrong_sentence).cpu().numpy().squeeze()
                    wrong_lang_tensor = torch.tensor(wrong_embedding, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        wrong_score = model.score_pairs(motion_tensor, homography_tensor, wrong_lang_tensor)
                        wrong_score = torch.sigmoid(wrong_score).item()
                    
                    non_gt_scores.append(wrong_score)
    
    gt_avg = np.mean(gt_scores) if gt_scores else 0.0
    non_gt_avg = np.mean(non_gt_scores) if non_gt_scores else 0.0
    
    print(f"  GT pairs: {len(gt_scores)}, avg score: {gt_avg:.4f}")
    print(f"  Non-GT pairs: {len(non_gt_scores)}, avg score: {non_gt_avg:.4f}")
    print(f"  Separation: {gt_avg - non_gt_avg:+.4f}")
    
    return gt_avg, non_gt_avg, len(gt_scores), len(non_gt_scores)


def main():
    # Configuration
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    
    kitti_root = "refer-kitti/KITTI"
    refer_kitti_root = "refer-kitti"
    test_sequence = "0011"
    weights_path = "gmc_link_with_homography_weights.pth"
    
    # Load model
    print(f"\nLoading model from {weights_path}...")
    model = MotionLanguageAlignerWithHomography(
        motion_dim=8,
        homography_dim=5,
        lang_dim=384,
        embed_dim=256
    ).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✓ Model loaded")
    
    # Load text encoder
    print("\nLoading text encoder...")
    encoder = TextEncoder(device=str(device))
    print("✓ Text encoder loaded")
    
    # Evaluate
    gt_avg, non_gt_avg, num_gt, num_non_gt = evaluate_sequence(
        model, encoder, test_sequence, kitti_root, refer_kitti_root, device=device
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Sequence: {test_sequence}")
    print(f"GT avg score: {gt_avg:.4f} ({num_gt} pairs)")
    print(f"Non-GT avg score: {non_gt_avg:.4f} ({num_non_gt} pairs)")
    print(f"Separation: {gt_avg - non_gt_avg:+.4f}")
    print("=" * 60)
    print("\nBaseline (Exp 17/20 - Geometric preprocessing):")
    print("  GT avg score: 0.5446")
    print("  Non-GT avg score: 0.2922")
    print("  Separation: +0.2524")
    print("=" * 60)


if __name__ == "__main__":
    main()
