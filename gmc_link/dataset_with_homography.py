"""
Dataset generation for Learning Camera Motion approach.

Instead of warping coordinates with homography, this dataset:
1. Uses IMAGE-FRAME motion vectors (not world-frame)
2. Extracts homography FEATURES as additional input
3. Lets the model learn to compensate for camera ego-motion
"""
import json
import os
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset

# Import base dataset utilities
from gmc_link.dataset import (
    load_refer_kitti_expressions,
    load_labels_with_ids,
    load_kitti_tracking_labels,
    is_motion_expression,
    _collect_expressions,
    _extract_target_centroids,
    MOTION_KEYWORDS,
)
from gmc_link.utils import VELOCITY_SCALE
from gmc_link.alignment_with_homography import decompose_homography_features
from gmc_link.core import ORBHomographyEngine
import cv2

HOMOGRAPHY_CACHE = {}


class MotionLanguageWithHomographyDataset(Dataset):
    """
    PyTorch Dataset for learning camera motion compensation.
    Each sample: (motion_vector, homography_features, language_embedding, label)
    
    Motion vector: 8D in IMAGE-FRAME (not compensated)
    Homography features: 5D [tx, ty, sx, sy, theta]
    """
    
    def __init__(self, motion_data, homography_data, language_data, labels):
        assert len(motion_data) == len(homography_data) == len(language_data) == len(labels)
        self.motion_data = motion_data
        self.homography_data = homography_data
        self.language_data = language_data
        self.labels = labels
    
    def __len__(self):
        return len(self.motion_data)
    
    def __getitem__(self, idx):
        motion = torch.tensor(self.motion_data[idx], dtype=torch.float32)
        homography = torch.tensor(self.homography_data[idx], dtype=torch.float32)
        lang = torch.tensor(self.language_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return motion, homography, lang, label


def collate_fn_with_homography(batch):
    """Stack into batches with homography features."""
    motion_batch = torch.stack([item[0] for item in batch], dim=0)
    homography_batch = torch.stack([item[1] for item in batch], dim=0)
    language_batch = torch.stack([item[2] for item in batch], dim=0)
    label_batch = torch.stack([item[3] for item in batch], dim=0)
    return motion_batch, homography_batch, language_batch, label_batch


def _generate_bce_pairs_with_homography(
    track_centroids: Dict[int, Dict[int, Tuple[float, float, float, float]]],
    sentence: str,
    embedding: np.ndarray,
    all_sentences: List[str],
    sentence_embeddings: Dict[str, np.ndarray],
    frame_gap: int,
    frame_shape: Tuple[int, int],
    seq: str = None,
    frame_dir: str = None,
    orb_engine: Any = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Generate pairs with:
    - Motion vectors in IMAGE-FRAME (not world-frame)
    - Homography features (5D geometric representation)
    - Language embeddings
    - Labels
    
    Key difference from original: NO warping of coordinates!
    """
    h, w = frame_shape
    motion_data = []
    homography_data = []
    language_data = []
    labels = []
    
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
            
            # NEW: Compute homography FIRST, then warp coordinates to world-frame
            if frame_dir is not None and orb_engine is not None and seq is not None:
                cache_key = (seq, curr_fid, future_fid)
                if cache_key in HOMOGRAPHY_CACHE:
                    homography = HOMOGRAPHY_CACHE[cache_key]
                else:
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
                    HOMOGRAPHY_CACHE[cache_key] = homography
            else:
                homography = None
            
            # Extract centroids in image-frame
            cx1, cy1, bw1, bh1 = centroids[curr_fid]
            cx2, cy2, bw2, bh2 = centroids[future_fid]
            
            # WARP coordinates to world-frame (camera-compensated)
            if homography is not None:
                cx1_world, cy1_world = warp_point(cx1, cy1, homography)
                cx2_world, cy2_world = warp_point(cx2, cy2, homography)
                bw1_world, bh1_world = bw1, bh1  # Size doesn't change much
                bw2_world, bh2_world = bw2, bh2
            else:
                # No homography - use image-frame as fallback
                cx1_world, cy1_world = cx1, cy1
                cx2_world, cy2_world = cx2, cy2
                bw1_world, bh1_world = bw1, bh1
                bw2_world, bh2_world = bw2, bh2
            
            # Add synthetic jitter for robustness (in world-frame)
            j_cx2 = cx2_world + np.random.uniform(-2.0, 2.0)
            j_cy2 = cy2_world + np.random.uniform(-2.0, 2.0)
            j_bw2 = bw2_world + np.random.uniform(-2.0, 2.0)
            j_bh2 = bh2_world + np.random.uniform(-2.0, 2.0)
            
            # WORLD-FRAME velocity (TRUE object motion, camera-compensated!)
            dx = (j_cx2 - cx1_world) / w * VELOCITY_SCALE
            dy = (j_cy2 - cy1_world) / h * VELOCITY_SCALE
            dw = (j_bw2 - bw1_world) / w * VELOCITY_SCALE
            dh = (j_bh2 - bh1_world) / h * VELOCITY_SCALE
            
            cx_n, cy_n = cx1_world / w, cy1_world / h
            bw_n, bh_n = bw1_world / w, bh1_world / h
            
            motion_vec = np.array(
                [dx, dy, dw, dh, cx_n, cy_n, bw_n, bh_n], dtype=np.float32
            )
            
            
            # Extract homography features (homography already computed above)
            # Extract homography features (homography already computed above)
            if homography is not None:
                # Extract full 8D homography parameters
                h_features = decompose_homography_features(homography)
                # Normalize translation components by image size for scale invariance
                h_features[2] /= w  # h13 (tx)
                h_features[5] /= h  # h23 (ty)
            else:
                # Identity homography (no camera motion)
                h_features = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity: I = [[1,0,0],[0,1,0],[0,0,1]]
                # No homography available - use identity
                h_features = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity: I = [[1,0,0],[0,1,0],[0,0,1]]
            
            # === Positive pair ===
            motion_data.append(motion_vec)
            homography_data.append(h_features)
            language_data.append(embedding)
            labels.append(1.0)
            
            # === Negative pairs ===
            # 1. Wrong sentence
            wrong_sentence = random.choice([s for s in all_sentences if s != sentence])
            wrong_embedding = sentence_embeddings[wrong_sentence]
            motion_data.append(motion_vec)
            homography_data.append(h_features)
            language_data.append(wrong_embedding)
            labels.append(0.0)
            
            # 2. Zero velocity (stationary) with correct sentence
            zero_motion = np.array([0.0, 0.0, 0.0, 0.0, cx_n, cy_n, bw_n, bh_n], dtype=np.float32)
            motion_data.append(zero_motion)
            homography_data.append(h_features)
            language_data.append(embedding)
            labels.append(0.0)
            
            # 3. Inverted velocity with correct sentence
            inv_motion = np.array([-dx, -dy, -dw, -dh, cx_n, cy_n, bw_n, bh_n], dtype=np.float32)
            motion_data.append(inv_motion)
            homography_data.append(h_features)
            language_data.append(embedding)
            labels.append(0.0)
    
    return motion_data, homography_data, language_data, labels


def build_training_data_with_homography(
    sequences: List[str],
    kitti_root: str,
    refer_kitti_root: str,
    sentence_embeddings: Dict[str, np.ndarray],
    frame_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training data with homography features for learning camera motion.
    
    Returns:
        motion_data: (N, 8) - IMAGE-FRAME motion vectors
        homography_data: (N, 5) - Homography features [tx, ty, sx, sy, theta]
        language_data: (N, lang_dim) - Language embeddings
        labels: (N,) - Binary labels
    """
    orb_engine = ORBHomographyEngine(max_features=1500)
    
    all_motion = []
    all_homography = []
    all_language = []
    all_labels = []
    
    for seq in sequences:
        print(f"Processing sequence {seq}...")
        
        # Load data
        expr_dir = os.path.join(refer_kitti_root, "expression", seq)
        labels_dir = os.path.join(refer_kitti_root, "KITTI", "labels_with_ids", "image_02", seq)
        frame_dir = os.path.join(kitti_root, "training", "image_02", seq)
        
        expressions = load_refer_kitti_expressions(expr_dir)
        motion_expressions = [e for e in expressions if is_motion_expression(e["sentence"])]
        
        all_sentences = [e["sentence"] for e in motion_expressions]
        
        labels = load_labels_with_ids(labels_dir)
        frame_shape = (375, 1242)  # KITTI image size
        
        # Process each expression
        for expr in motion_expressions:
            sentence = expr["sentence"]
            embedding = sentence_embeddings[sentence]
            
            # Extract target track centroids
            label_map = expr["label"]
            track_centroids = _extract_target_centroids(
                refer_kitti_root, seq, label_map, frame_shape=frame_shape
            )
            
            # Generate pairs WITH homography features
            motion, homography, language, label = _generate_bce_pairs_with_homography(
                track_centroids,
                sentence,
                embedding,
                all_sentences,
                sentence_embeddings,
                frame_gap,
                frame_shape,
                seq=seq,
                frame_dir=frame_dir,
                orb_engine=orb_engine,
            )
            
            all_motion.extend(motion)
            all_homography.extend(homography)
            all_language.extend(language)
            all_labels.extend(label)
    
    print(f"Total samples generated: {len(all_motion)}")
    print(f"Positive samples: {sum(all_labels)}")
    print(f"Negative samples: {len(all_labels) - sum(all_labels)}")
    
    return (
        np.array(all_motion, dtype=np.float32),
        np.array(all_homography, dtype=np.float32),
        np.array(all_language, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
    )


def warp_point(x: float, y: float, H: np.ndarray) -> tuple:
    """
    Warp a single point using homography matrix.
    
    Args:
        x, y: Point coordinates in pixels
        H: (3, 3) homography matrix
    
    Returns:
        (x_warped, y_warped): Warped coordinates
    """
    if H is None:
        return x, y
    
    # Create homogeneous coordinates
    point = np.array([[x, y]], dtype=np.float32).reshape(1, 1, 2)
    
    # Warp using cv2
    point_warped = cv2.perspectiveTransform(point, H)
    
    return float(point_warped[0, 0, 0]), float(point_warped[0, 0, 1])
