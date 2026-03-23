import logging
import os

import torch
import numpy as np
from PIL import Image

import folder_paths

logger = logging.getLogger(__name__)


class RSFrameCollector:
    """Collect edited frames to disk, reassemble into full video batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": 0}),
                "total_frames": ("INT", {"default": 1, "min": 1}),
                "original_frame_count": ("INT", {"default": 97, "min": 1}),
                "every_nth": ("INT", {"default": 8, "min": 1}),
                "start_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "prefix": ("STRING", {"default": "edited_frame"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "sequence")
    FUNCTION = "execute"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _get_dir(self, prefix: str) -> str:
        base = folder_paths.get_output_directory()
        path = os.path.join(base, "frame_collector", prefix)
        os.makedirs(path, exist_ok=True)
        return path

    def execute(self, source_video, index: int, total_frames: int,
                original_frame_count: int, every_nth: int, start_index: int,
                image=None, prefix: str = "edited_frame"):
        save_dir = self._get_dir(prefix)

        # Save current frame to disk if connected
        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            filepath = os.path.join(save_dir, f"{prefix}_{index:04d}.png")
            Image.fromarray(img_np).save(filepath)
            logger.info(f"RSFrameCollector: saved frame {index} to {filepath}")

        # Target resolution from source video
        target_h, target_w = source_video.shape[1], source_video.shape[2]

        # Build full video-length sequence
        all_frame_indices = list(range(0, original_frame_count, every_nth))
        output_frames = []

        for ki, fi in enumerate(all_frame_indices):
            if ki + 1 < len(all_frame_indices):
                next_fi = all_frame_indices[ki + 1]
            else:
                next_fi = original_frame_count
            count = max(next_fi - fi, 1)

            edited_path = os.path.join(save_dir, f"{prefix}_{ki:04d}.png")
            if os.path.exists(edited_path):
                img = Image.open(edited_path).convert("RGB")
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                output_frames.extend([t] * count)
            else:
                for f in range(fi, min(fi + count, original_frame_count)):
                    output_frames.append(source_video[f])

        # Ensure exactly original_frame_count frames
        if len(output_frames) < original_frame_count:
            output_frames.extend([output_frames[-1]] * (original_frame_count - len(output_frames)))
        elif len(output_frames) > original_frame_count:
            output_frames = output_frames[:original_frame_count]

        sequence = torch.stack(output_frames, dim=0)
        collected_count = sum(
            1 for i in range(start_index, start_index + total_frames)
            if os.path.exists(os.path.join(save_dir, f"{prefix}_{i:04d}.png"))
        )
        logger.info(f"RSFrameCollector: {collected_count}/{total_frames} collected, sequence {sequence.shape[0]} frames")

        current = image if image is not None else sequence[:1]
        return (current, sequence)
