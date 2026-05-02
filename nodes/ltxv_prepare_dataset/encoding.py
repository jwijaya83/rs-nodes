"""Text + latent encoding for the prepare-dataset node.

In-process encoders that use ComfyUI's already-loaded CLIP and VAE
nodes. Skips files already on disk (idempotent). For audio_latents,
the orchestrator currently still falls back to subprocess; an
audio_vae node input + in-process encoder is a planned follow-up.
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

import comfy.utils

from .dataset_io import (
    condition_path_for_clip,
    latent_path_for_clip,
    normalize_loaded_entries,
    COND_TOKEN_LIMIT,
)

# captioning is imported lazily inside encode_conditions_inprocess to avoid
# circular import at module-load time
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def encode_conditions_inprocess(clip, dataset_json_path: Path, conditions_dir: Path,
                                 character_names=None):
    """Encode captions in-process using ComfyUI's already-loaded CLIP text encoder.

    ComfyUI's LTX CLIP runs Gemma + feature extractor (blocks 1+2 of the text
    encoder pipeline). This produces the same output as process_captions.py's
    text_encoder.encode() + embeddings_processor.feature_extractor() but without
    loading Gemma from disk.

    LTX-2   (single_linear): cond is [B, seq, 3840] — video and audio share features.
    LTX-2.3 (dual_linear):   cond is [B, seq, 6144] — split at 4096 for video/audio.
    """
    from .captioning import normalize_caption_for_encode  # avoid circular import

    conditions_dir = Path(conditions_dir)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_json_path) as f:
        entries = json.load(f)

    data_root = dataset_json_path.parent
    # Normalize relative media_paths to absolute so downstream path
    # logic is consistent — without this, _condition_path_for_clip
    # used to fall back to a flat <stem>.pt root path and the encoder
    # would write to conditions/foo.pt instead of conditions/clips/foo.pt,
    # corrupting the dataset layout. _condition_path_for_clip is now
    # robust to relative paths too, but normalizing here means the
    # encoder works correctly even if that helper changes.
    normalize_loaded_entries(entries, data_root)
    to_encode = []
    for entry in entries:
        media_path = Path(entry["media_path"])
        # Reuse the shared path helper so audit / encoder always look
        # at exactly the same condition file for a given clip.
        output_file = condition_path_for_clip(data_root, media_path)
        if not output_file.exists():
            to_encode.append((entry, output_file))

    if not to_encode:
        logger.info("All conditions already exist, skipping text encoding")
        return

    logger.info(f"Encoding {len(to_encode)} captions in-process with ComfyUI CLIP...")
    pbar = comfy.utils.ProgressBar(len(to_encode))

    for i, (entry, output_file) in enumerate(to_encode):
        caption = entry.get("caption", "")
        caption = normalize_caption_for_encode(caption, character_names)

        tokens = clip.tokenize(caption)
        result = clip.encode_from_tokens(tokens, return_dict=True)
        cond = result["cond"]  # [1, seq_len, proj_dim]

        proj_dim = cond.shape[2]
        if proj_dim == 3840:
            # LTX-2: same features for both modalities
            video_features = cond
            audio_features = cond
        elif proj_dim >= 6144:
            # LTX-2.3: split at 4096 for separate video/audio connectors
            video_features = cond[:, :, :4096]
            audio_features = cond[:, :, 4096:]
        else:
            video_features = cond
            audio_features = cond

        orig_seq_len = cond.shape[1]

        # EmbeddingsProcessor.create_embeddings (block 3, applied during training)
        # requires seq_len divisible by num_learnable_registers (128).
        # ComfyUI's CLIP doesn't pad like Gemma's tokenizer, so we left-pad here.
        pad_to = 128
        pad_len = (pad_to - orig_seq_len % pad_to) % pad_to
        if pad_len > 0:
            # Left-pad features with zeros (matching Gemma's left-padding convention)
            video_features = torch.nn.functional.pad(video_features, (0, 0, pad_len, 0))
            if audio_features is not video_features:
                audio_features = torch.nn.functional.pad(audio_features, (0, 0, pad_len, 0))

        padded_len = orig_seq_len + pad_len
        # Attention mask: False for padding (left), True for real tokens (right)
        attention_mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.bool),
            torch.ones(orig_seq_len, dtype=torch.bool),
        ]) if pad_len > 0 else torch.ones(orig_seq_len, dtype=torch.bool)

        embedding_data = {
            "video_prompt_embeds": video_features[0].cpu().contiguous(),
            "prompt_attention_mask": attention_mask.cpu().contiguous(),
        }
        if proj_dim >= 6144:
            embedding_data["audio_prompt_embeds"] = audio_features[0].cpu().contiguous()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding_data, output_file)
        pbar.update_absolute(i + 1, len(to_encode))

    logger.info(f"Encoded {len(to_encode)} captions in-process")


def encode_latents_inprocess(
    vae, dataset_json_path: Path, latents_dir: Path, existing_entries: list[dict],
    target_w: int = 0, target_h: int = 0,
):
    """Encode video/image clips in-process using ComfyUI's already-loaded VAE.

    ComfyUI's VAE.encode() handles device management, batching, and normalisation.
    Input format expected: [F, H, W, C] float [0,1] for video, [1, H, W, C] for images.
    Output: 5-D latent [1, C, F', H', W'] from 3D video VAE.
    """
    latents_dir = Path(latents_dir)
    latents_dir.mkdir(parents=True, exist_ok=True)

    data_root = dataset_json_path.parent
    to_encode = []
    for entry in existing_entries:
        media_path = Path(entry["media_path"])
        output_file = latent_path_for_clip(data_root, media_path)
        if not output_file.exists():
            to_encode.append((entry, media_path, output_file))

    if not to_encode:
        logger.info("All latents already exist, skipping VAE encoding")
        return

    logger.info(f"Encoding {len(to_encode)} clips in-process with ComfyUI VAE...")
    pbar = comfy.utils.ProgressBar(len(to_encode))

    for i, (entry, media_path, output_file) in enumerate(to_encode):
        print(f"[VAE encode {i+1}/{len(to_encode)}] {media_path.name}")
        ext = media_path.suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            # Read image as [1, H, W, 3] float [0,1]
            img = cv2.imread(str(media_path))
            if img is None:
                logger.warning(f"Could not read image: {media_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = torch.from_numpy(img).float() / 255.0
            pixels = pixels.unsqueeze(0)  # [1, H, W, 3]
            fps = 1.0
        else:
            pixels, fps = load_video_frames(media_path)
            if pixels is None:
                logger.warning(f"Could not read video: {media_path}")
                continue

        # Resize to target resolution if needed (allows re-encoding at different res
        # without re-extracting clips). Images keep native resolution (tiny tensors).
        if target_w > 0 and target_h > 0 and pixels.shape[0] > 1:
            _, cur_h, cur_w, _ = pixels.shape
            if cur_w != target_w or cur_h != target_h:
                # Aspect-preserving scale + letterbox (handles full_frame clips
                # that may have different aspect ratios than target)
                scale = min(target_w / cur_w, target_h / cur_h)
                new_w = int(cur_w * scale)
                new_h = int(cur_h * scale)
                scaled = torch.nn.functional.interpolate(
                    pixels.permute(0, 3, 1, 2),  # [F, C, H, W]
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                if new_w == target_w and new_h == target_h:
                    pixels = scaled.permute(0, 2, 3, 1)
                else:
                    # Pad to exact target size with black
                    padded = torch.zeros(pixels.shape[0], 3, target_h, target_w,
                                         dtype=scaled.dtype, device=scaled.device)
                    pad_y = (target_h - new_h) // 2
                    pad_x = (target_w - new_w) // 2
                    padded[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
                    pixels = padded.permute(0, 2, 3, 1)  # [F, H, W, C]

        # VAE.encode() expects [F, H, W, C] and returns [1, C, F', H', W']
        latents = vae.encode(pixels)

        if latents.ndim == 5:
            latents_save = latents[0]  # [C, F', H', W']
        else:
            latents_save = latents

        _, num_frames_lat, h_lat, w_lat = latents_save.shape

        latent_data = {
            "latents": latents_save.cpu().contiguous(),
            "num_frames": num_frames_lat,
            "height": h_lat,
            "width": w_lat,
            "fps": fps,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent_data, output_file)
        pbar.update_absolute(i + 1, len(to_encode))

    logger.info(f"Encoded {len(to_encode)} clips in-process")


def load_video_frames(video_path: Path):
    """Load all frames from a video as [F, H, W, 3] float [0,1] tensor.
    Returns (tensor, fps) on success, or (None, 0) on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        return None, 0

    video = np.stack(frames)  # [F, H, W, 3] uint8
    video = torch.from_numpy(video).float() / 255.0
    return video, fps


def resolve_resolution_buckets(resolution_buckets: str, entries: list[dict]) -> str:
    """Add 1-frame image buckets derived from existing video buckets.

    Uses the spatial dimensions of each video bucket with frames=1,
    so images are resized/cropped to match the video resolution.
    This prevents the bucketing algorithm from creating a native-
    resolution image bucket that hijacks video clips due to a closer
    aspect ratio match.
    """
    has_images = any(
        entry.get("media_path", "").lower().endswith((".png", ".jpg", ".jpeg"))
        for entry in entries
    )
    if not has_images:
        return resolution_buckets

    existing_buckets = set(resolution_buckets.split(";"))
    for bucket_str in list(existing_buckets):
        parts = bucket_str.strip().split("x")
        if len(parts) >= 3 and int(parts[2]) > 1:
            image_bucket = f"{parts[0]}x{parts[1]}x1"
            if image_bucket not in existing_buckets:
                resolution_buckets = f"{resolution_buckets};{image_bucket}"
                existing_buckets.add(image_bucket)
                logger.info(f"Added image resolution bucket: {image_bucket}")
    return resolution_buckets
