"""Depth Anything V2 metric-outdoor-large wrapper. One-shot per-frame inference.

Output: float32 metric depth map, meters, shape (H, W) matching input image.

Set DEPTH_ANYTHING_LOCAL_DIR to load from a local snapshot dir (containing
config.json, preprocessor_config.json, model.safetensors). Falls back to HF Hub
identifier when env var unset.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

_HF_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def _resolve_model_path() -> str:
    return os.environ.get("DEPTH_ANYTHING_LOCAL_DIR", _HF_MODEL)


class DepthExtractor:
    def __init__(self, device: str = "cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        path = _resolve_model_path()
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(path, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        H, W = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth  # (1, h, w)
        pred = (
            torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(0)
        )
        return pred.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_batch(self, images_rgb: list[np.ndarray]) -> list[np.ndarray]:
        return [self.extract(im) for im in images_rgb]
