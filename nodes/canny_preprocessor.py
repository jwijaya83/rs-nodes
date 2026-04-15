"""Canny edge detection preprocessor for IC-LoRA structural control.

Takes any image, computes the closest LTXV-safe resolution (divisible by 128)
preserving aspect ratio, runs Canny edge detection, and returns the edge
images with the computed dimensions — ready to wire into RSLTXVICLoRAGuider.
"""

import logging
import math

import cv2
import numpy as np
import torch
import comfy.utils

logger = logging.getLogger(__name__)


class RSCannyPreprocessor:
    """Canny edge preprocessor with automatic LTXV-safe resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "width":          ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "height":         ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "low_threshold":  ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "clahe_clip":     ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                             "tooltip": "CLAHE adaptive contrast. Equalizes dark/bright regions for even edge density. 0 = disabled."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process"
    CATEGORY = "rs-nodes"

    def process(self, image, width=768, height=512,
                low_threshold=100, high_threshold=200, clahe_clip=2.0):
        # image is [B, H, W, C] float32 0-1
        src_h, src_w = image.shape[1], image.shape[2]
        aspect = src_w / src_h

        # Compute closest 128-aligned resolution from pixel budget
        # (128 = 32 VAE stride × 2 downscale_factor × 2 upscale halving)
        total_pixels = width * height
        target_h = int(math.sqrt(total_pixels / aspect))
        target_w = int(target_h * aspect)
        target_w = max(32, round(target_w / 32) * 32)
        target_h = max(32, round(target_h / 32) * 32)

        logger.info(f"{src_w}x{src_h} → {target_w}x{target_h}")

        # Resize to target resolution
        resized = comfy.utils.common_upscale(
            image.movedim(-1, 1), target_w, target_h, "bilinear", "center"
        ).movedim(1, -1)

        # Canny edge detection per frame
        num_frames = resized.shape[0]
        pbar = comfy.utils.ProgressBar(num_frames)
        frames = []
        for i in range(num_frames):
            frame_np = (resized[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            if clahe_clip > 0:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            # Convert single-channel edges to 3-channel
            edges_rgb = np.stack([edges, edges, edges], axis=-1)
            frames.append(torch.from_numpy(edges_rgb.astype(np.float32) / 255.0))
            pbar.update(1)

        result = torch.stack(frames).to(image.device)
        return (result, target_w, target_h)
