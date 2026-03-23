import json
import logging
import math
import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import folder_paths

logger = logging.getLogger(__name__)

_STATE_FILENAME = "frame_splitter_state.json"


def _state_path() -> str:
    return os.path.join(folder_paths.get_output_directory(), _STATE_FILENAME)


def _load_index() -> int:
    path = _state_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f).get("index", 0)
        except (json.JSONDecodeError, OSError):
            pass
    return 0


def _save_index(index: int) -> None:
    with open(_state_path(), "w") as f:
        json.dump({"index": index}, f)


def _build_index_sheet(images, all_frame_indices, start_index, collector_dir, prefix):
    """Build a grid of all extracted frames with index labels.
    Frames before start_index are dimmed. Edited frames show the edited version."""
    total = len(all_frame_indices)
    cols = min(total, 4)
    rows = math.ceil(total / cols)

    # Get source size then scale down for thumbnails
    first_np = (images[all_frame_indices[0]].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    src_w, src_h = Image.fromarray(first_np).size
    # Target ~320px wide per thumbnail
    thumb_w = 320
    scale = thumb_w / src_w
    fw = thumb_w
    fh = int(src_h * scale)

    # Build original and edited frame pairs
    originals = []
    edited = []
    for abs_i, fi in enumerate(all_frame_indices):
        img_np = (images[fi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        orig = Image.fromarray(img_np).resize((fw, fh), Image.LANCZOS)
        originals.append(orig)

        edited_path = os.path.join(collector_dir, f"{prefix}_{abs_i:04d}.png")
        if os.path.exists(edited_path):
            img = Image.open(edited_path).convert("RGB")
            if img.size != (fw, fh):
                img = img.resize((fw, fh), Image.LANCZOS)
            edited.append(img)
        else:
            edited.append(None)

    # Font
    font_size = max(fh // 8, 20)
    font = None
    for font_path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default(size=font_size)

    # Each cell is a side-by-side pair (2*fw wide, fh tall)
    cell_w = fw * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * fh), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    for i in range(total):
        col = i % cols
        row = i // cols
        x, y = col * cell_w, row * fh

        orig = originals[i]
        edit = edited[i]

        if i < start_index:
            # Dim skipped frames
            dimmed = Image.blend(orig, Image.new("RGB", orig.size, (0, 0, 0)), 0.7)
            sheet.paste(dimmed, (x, y))
            if edit:
                dimmed_edit = Image.blend(edit, Image.new("RGB", edit.size, (0, 0, 0)), 0.7)
                sheet.paste(dimmed_edit, (x + fw, y))
            else:
                sheet.paste(dimmed, (x + fw, y))
        else:
            sheet.paste(orig, (x, y))
            if edit:
                sheet.paste(edit, (x + fw, y))

        # Draw index label
        label = str(i)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 8
        color = (100, 100, 100) if i < start_index else (255, 255, 0)
        draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=(0, 0, 0))
        draw.text((x + pad, y + pad), label, fill=color, font=font)

    # Convert back to tensor [1, H, W, C]
    sheet_np = np.array(sheet).astype(np.float32) / 255.0
    return torch.from_numpy(sheet_np).unsqueeze(0)


class RSFrameSplitter:
    """Extract every Nth frame from a video batch, one per queue run.
    Built-in counter auto-increments each run. Reset to start over."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "every_nth": ("INT", {"default": 8, "min": 1}),
                "start_index": ("INT", {"default": 0, "min": 0}),
                "override_index": ("INT", {"default": -1, "min": -1}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "edited_frame"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "index", "total_frames", "index_sheet", "previous_image")
    FUNCTION = "execute"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, images, every_nth: int, start_index: int, override_index: int,
                reset: bool, prefix: str = "edited_frame"):
        num_frames = images.shape[0]
        all_frame_indices = list(range(0, num_frames, every_nth))
        frame_indices = all_frame_indices[start_index:]
        total = len(frame_indices)

        collector_dir = os.path.join(folder_paths.get_output_directory(), "frame_collector", prefix)

        if override_index >= 0:
            abs_idx = min(override_index, len(all_frame_indices) - 1)
        else:
            index = _load_index()
            if reset:
                index = 0
                _save_index(0)
            idx = min(index, total - 1)
            abs_idx = start_index + idx

        image = images[all_frame_indices[abs_idx]].unsqueeze(0)
        index_sheet = _build_index_sheet(images, all_frame_indices, start_index, collector_dir, prefix)

        # Load previous edited frame from collector's disk folder
        prev_abs = abs_idx - 1
        previous_image = image  # fallback: current frame if no previous exists
        if prev_abs >= 0:
            prev_path = os.path.join(collector_dir, f"{prefix}_{prev_abs:04d}.png")
            if os.path.exists(prev_path):
                prev_img = Image.open(prev_path).convert("RGB")
                previous_image = torch.from_numpy(
                    np.array(prev_img).astype(np.float32) / 255.0
                ).unsqueeze(0)
                logger.info(f"RSFrameSplitter: loaded previous frame {prev_abs}")

        logger.info(f"RSFrameSplitter: frame {abs_idx}/{len(all_frame_indices)}")

        # Advance counter at the end
        if override_index < 0 and not reset:
            _save_index(idx + 1)

        return (image, abs_idx, total, index_sheet, previous_image)
