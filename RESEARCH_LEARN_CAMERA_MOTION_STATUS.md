# Research: Learning Camera Motion - Implementation Status

## Progress Tracker

### ✅ Step 1: Architecture Implemented
- File: `gmc_link/alignment_with_homography.py`
- `MotionLanguageAlignerWithHomography` class
- `decompose_homography_features()` function
- Motion encoder (8D) + Homography encoder (5D) + Fusion layer

### ✅ Step 2: Dataset with Homography Features
- File: `gmc_link/dataset_with_homography.py`
- `MotionLanguageWithHomographyDataset` class
- `build_training_data_with_homography()` function
- Extracts IMAGE-FRAME motion (no warping)
- Extracts 5D homography features [tx, ty, sx, sy, theta]
- Same negative sampling: 4 negatives per positive

### ⬜ Step 3: Training Script (TODO)
- Create: `gmc_link/train_with_homography.py`
- Modify train loop to handle (motion, homography, language, label) tuples
- Use `collate_fn_with_homography()`
- Call `model.score_pairs(motion, homography, language)`

### ⬜ Step 4: Train on Refer-KITTI (TODO)
- Train sequences: 0015, 0016, 0018
- Test sequence: 0011
- Compare with geometric baseline (Exp 17/20)

### ⬜ Step 5: Ablation Study (TODO)
- Train WITHOUT homography features (motion only)
- Train WITH homography features
- Compare accuracy and separation

### ⬜ Step 6: Comparison with Baseline (TODO)
- Geometric baseline (Exp 20): GT 0.5446, Non-GT 0.2922
- Learning approach: TBD
- Analyze learned camera compensation

## Quick Start for Step 3

```python
# gmc_link/train_with_homography.py (minimal template)
from gmc_link.alignment_with_homography import MotionLanguageAlignerWithHomography
from gmc_link.dataset_with_homography import (
    MotionLanguageWithHomographyDataset,
    collate_fn_with_homography,
    build_training_data_with_homography
)
from gmc_link.text_utils import TextEncoder

# Initialize model
model = MotionLanguageAlignerWithHomography(
    motion_dim=8,
    homography_dim=5,
    lang_dim=384,
    embed_dim=256
)

# Build dataset
encoder = TextEncoder(device="cuda")
# ... encode all sentences ...

motion_data, homography_data, language_data, labels = build_training_data_with_homography(
    sequences=["0015", "0016", "0018"],
    kitti_root="refer-kitti/KITTI",
    refer_kitti_root="refer-kitti",
    sentence_embeddings=embeddings,
    frame_gap=5
)

# Create dataset
dataset = MotionLanguageWithHomographyDataset(
    motion_data, homography_data, language_data, labels
)

# Train loop
for epoch in range(epochs):
    for motion, homography, lang, labels in dataloader:
        scores = model.score_pairs(motion, homography, lang)
        loss = loss_fn(scores, labels)
        # ... backward pass ...
```

## Next Actions

To continue this research:

1. **Create training script** (copy from `train.py`, modify for homography)
2. **Run training** on Refer-KITTI sequences
3. **Evaluate** on sequence 0011
4. **Compare** with Exp 20 baseline
5. **Document** in RESEARCH_NOTES.md as new experiment

## Expected Outcomes

**If successful:**
- Model learns to compensate for camera motion from data
- More robust to bad homographies than geometric preprocessing
- Competitive or better accuracy than geometric baseline

**If unsuccessful:**
- Insufficient training data diversity (camera motions)
- Model cannot learn effective compensation
- Falls back to: use geometric preprocessing (proven approach)

## Files Created

- `gmc_link/alignment_with_homography.py` - Architecture ✓
- `gmc_link/dataset_with_homography.py` - Dataset ✓
- `gmc_link/train_with_homography.py` - Training TODO
- This file - Status tracker ✓

Branch: `research/learn-camera-motion`
