"""Canny edge detection preprocessor for IC-LoRA structural control.

Takes any image, computes the closest LTXV-safe resolution (divisible by 128)
preserving aspect ratio, runs Canny edge detection, and returns the edge
images with the computed dimensions — ready to wire into RSLTXVICLoRAGuider.
"""

import math

import cv2
import numpy as np
import torch
import comfy.utils


class RSCannyPreprocessor:
    """Canny edge preprocessor with automatic LTXV-safe resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "width":          ("INT", {"default": 768, "min": 128, "max": 8192, "step": 128}),
                "height":         ("INT", {"default": 512, "min": 128, "max": 8192, "step": 128}),
                "low_threshold":  ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process"
    CATEGORY = "rs-nodes"

    def process(self, image, width=768, height=512,
                low_threshold=100, high_threshold=200):
        # image is [B, H, W, C] float32 0-1
        src_h, src_w = image.shape[1], image.shape[2]
        aspect = src_w / src_h

        # Compute closest 128-aligned resolution from pixel budget
        # (128 = 32 VAE stride × 2 downscale_factor × 2 upscale halving)
        total_pixels = width * height
        target_h = int(math.sqrt(total_pixels / aspect))
        target_w = int(target_h * aspect)
        target_w = max(128, round(target_w / 128) * 128)
        target_h = max(128, round(target_h / 128) * 128)

        print(f"[RSCannyPreprocessor] {src_w}x{src_h} → {target_w}x{target_h}")

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
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            # Convert single-channel edges to 3-channel
            edges_rgb = np.stack([edges, edges, edges], axis=-1)
            frames.append(torch.from_numpy(edges_rgb.astype(np.float32) / 255.0))
            pbar.update(1)

        result = torch.stack(frames).to(image.device)
        return (result, target_w, target_h)
