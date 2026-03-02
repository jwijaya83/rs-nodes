"""IC-LoRA guider for LTXV structural control (canny/depth/pose).

Loads an IC-LoRA (union or individual), encodes control images as guide
latents with optional dilation, and produces a GUIDER that plugs into
RSLTXVGenerate's guider input. Supports full multimodal (audio-video).

The union model (ltx-2-19b-ic-lora-union-control-ref0.5.safetensors)
automatically detects the control type from the visual pattern of the
preprocessed image — no control type selector is needed.
"""

import torch
import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from comfy_extras.nodes_lt import (
    preprocess as ltxv_preprocess,
    LTXVAddGuide,
)

from ..utils.multimodal_guider import MultimodalGuider


# ---------------------------------------------------------------------------
# IC-LoRA Guider — extends MultimodalGuider with control latent injection
# ---------------------------------------------------------------------------

class ICLoRAMultimodalGuider(MultimodalGuider):
    """MultimodalGuider that injects IC-LoRA control latents around sampling.

    Control latents are appended to the video temporal dimension before
    sampling and cropped back out after, so the model sees the structural
    guidance while the output contains only the generated content.

    Overrides sample() (not inner_sample()) because control latent appending
    must happen before CFGGuider.sample() packs NestedTensors.
    """

    def __init__(self, model, positive, negative,
                 control_latents, control_masks, num_control_frames,
                 **kwargs):
        super().__init__(model, positive, negative, **kwargs)
        self.control_latents = control_latents   # list of [1,C,T,H,W] on CPU
        self.control_masks = control_masks       # list of [1,1,T,H,W] on CPU
        self.num_control_frames = num_control_frames

    def sample(self, noise, latent_image, sampler, sigmas,
               denoise_mask=None, **kwargs):
        if self.num_control_frames == 0:
            return super().sample(noise, latent_image, sampler, sigmas,
                                  denoise_mask=denoise_mask, **kwargs)

        # --- 1. Unbind NestedTensor if AV model ---
        is_nested = hasattr(latent_image, 'is_nested') and latent_image.is_nested
        audio_latent = audio_noise = audio_mask = None

        if is_nested:
            import comfy.nested_tensor
            video, audio_latent = latent_image.unbind()
            video_noise, audio_noise = noise.unbind()
            if denoise_mask is not None and denoise_mask.is_nested:
                video_dmask, audio_mask = denoise_mask.unbind()
            else:
                video_dmask = denoise_mask
        else:
            video = latent_image
            video_noise = noise
            video_dmask = denoise_mask

        # --- 2. Append control latents to video temporal dim ---
        ctrl_parts = []
        mask_parts = []
        for ctrl_lat, ctrl_mask in zip(self.control_latents, self.control_masks):
            ctrl_parts.append(ctrl_lat.to(device=video.device, dtype=video.dtype))
            mask_parts.append(ctrl_mask.to(device=video.device, dtype=video.dtype))

        all_ctrl = torch.cat(ctrl_parts, dim=2)
        all_ctrl_mask = torch.cat(mask_parts, dim=2)

        # Pad control latent channels if video has more (AV channel concat)
        if video.shape[1] > all_ctrl.shape[1]:
            pad_len = video.shape[1] - all_ctrl.shape[1]
            all_ctrl = torch.nn.functional.pad(
                all_ctrl, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0
            )

        video = torch.cat([video, all_ctrl], dim=2)

        # --- 3. Extend noise with zeros for control positions ---
        ctrl_noise = torch.zeros(
            video_noise.shape[0], video_noise.shape[1],
            self.num_control_frames,
            video_noise.shape[3], video_noise.shape[4],
            device=video_noise.device, dtype=video_noise.dtype,
        )
        video_noise = torch.cat([video_noise, ctrl_noise], dim=2)

        # --- 4/5. Extend denoise_mask, handling spatial dim expansion ---
        if video_dmask is not None:
            target_h = max(video_dmask.shape[3], all_ctrl_mask.shape[3])
            target_w = max(video_dmask.shape[4], all_ctrl_mask.shape[4])
            if video_dmask.shape[3] == 1 or video_dmask.shape[4] == 1:
                video_dmask = video_dmask.expand(-1, -1, -1, target_h, target_w)
            if all_ctrl_mask.shape[3] == 1 or all_ctrl_mask.shape[4] == 1:
                all_ctrl_mask = all_ctrl_mask.expand(-1, -1, -1, target_h, target_w)
            video_dmask = torch.cat([video_dmask, all_ctrl_mask], dim=2)
        else:
            # No existing mask — create ones for content, append control mask
            content_frames = video.shape[2] - self.num_control_frames
            h, w = all_ctrl_mask.shape[3], all_ctrl_mask.shape[4]
            content_mask = torch.ones(
                1, 1, content_frames, h, w,
                device=video.device, dtype=video.dtype,
            )
            video_dmask = torch.cat([content_mask, all_ctrl_mask], dim=2)

        # --- 6. Recombine as NestedTensor if AV ---
        if is_nested:
            import comfy.nested_tensor
            latent_image = comfy.nested_tensor.NestedTensor((video, audio_latent))
            noise = comfy.nested_tensor.NestedTensor((video_noise, audio_noise))
            if audio_mask is not None:
                denoise_mask = comfy.nested_tensor.NestedTensor((video_dmask, audio_mask))
            else:
                denoise_mask = video_dmask
        else:
            latent_image = video
            noise = video_noise
            denoise_mask = video_dmask

        # --- 7. Run full MultimodalGuider sampling pipeline ---
        try:
            result = super().sample(noise, latent_image, sampler, sigmas,
                                    denoise_mask=denoise_mask, **kwargs)

            # --- 8/9/10. Crop control frames from result ---
            if hasattr(result, 'is_nested') and result.is_nested:
                import comfy.nested_tensor
                parts = result.unbind()
                video_out = parts[0][:, :, :-self.num_control_frames]
                result = comfy.nested_tensor.NestedTensor(
                    (video_out,) + tuple(parts[1:])
                )
            else:
                result = result[:, :, :-self.num_control_frames]

            return result
        finally:
            # Free control latents so VRAM is available for upscale pass
            self.control_latents = None
            self.control_masks = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RSLTXVICLoRAGuider:
    """Creates an IC-LoRA guider for LTXV structural control.

    Loads an IC-LoRA (union or individual), encodes control images (canny,
    depth, pose) as guide latents, and returns a GUIDER that plugs into
    RSLTXVGenerate's guider input. The union model automatically detects
    the control type from the visual pattern of the preprocessed image.

    Split responsibility: control images go through this guider; reference
    photos for pixel-perfect frame anchoring go through the generate node's
    first_image/last_image inputs (latent injection, independent of guider).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("MODEL",),
                "positive":  ("CONDITIONING",),
                "negative":  ("CONDITIONING",),
                "vae":       ("VAE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                # ── Control images ──
                "first_control":     ("IMAGE",),
                "last_control":      ("IMAGE",),
                # ── Strengths ──
                "lora_strength":     ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "control_strength":  ("FLOAT", {"default": 1.0, "min": 0.0,   "max": 1.0,  "step": 0.01}),
                # ── Dimensions ──
                "width":             ("INT",   {"default": 768, "min": 64,    "max": 8192, "step": 32}),
                "height":            ("INT",   {"default": 512, "min": 64,    "max": 8192, "step": 32}),
                "num_frames":        ("INT",   {"default": 97,  "min": 9,     "max": 8192, "step": 8}),
                # ── Guidance ──
                "cfg":               ("FLOAT", {"default": 3.0, "min": 0.0,   "max": 100.0, "step": 0.1}),
                "audio_cfg":         ("FLOAT", {"default": 7.0, "min": 0.0,   "max": 100.0, "step": 0.1}),
                "stg_scale":         ("FLOAT", {"default": 0.0, "min": 0.0,   "max": 10.0,  "step": 0.1}),
                "cfg_end":           ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "stg_end":           ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0,  "step": 0.1}),
                "stg_blocks":        ("STRING", {"default": "29"}),
                "rescale":           ("FLOAT", {"default": 0.7, "min": 0.0,   "max": 1.0,   "step": 0.01}),
                "modality_scale":    ("FLOAT", {"default": 1.0, "min": 0.0,   "max": 100.0, "step": 0.1}),
                # ── Misc ──
                "frame_rate":        ("FLOAT", {"default": 25.0, "min": 0.0,  "max": 1000.0, "step": 0.01}),
                "attention_mode":    (["auto", "default", "sage"],),
                "upscale":           ("BOOLEAN", {"default": False}),
                "crf":               ("INT",   {"default": 35,  "min": 0,     "max": 100}),
            },
        }

    RETURN_TYPES = ("GUIDER", "INT", "INT")
    RETURN_NAMES = ("guider", "width", "height")
    FUNCTION = "create_guider"
    CATEGORY = "rs-nodes"

    def create_guider(
        self, model, positive, negative, vae, lora_name,
        first_control=None, last_control=None,
        lora_strength=1.0, control_strength=1.0,
        width=768, height=512, num_frames=97,
        cfg=3.0, audio_cfg=7.0, stg_scale=0.0,
        cfg_end=-1.0, stg_end=-1.0, stg_blocks="29",
        rescale=0.7, modality_scale=1.0,
        frame_rate=25.0, attention_mode="auto",
        upscale=False, crf=35,
    ):
        m = model.clone()

        # --- Load IC-LoRA and extract metadata ---
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
        try:
            latent_downscale_factor = float(metadata["reference_downscale_factor"])
        except (KeyError, ValueError, TypeError):
            latent_downscale_factor = 1.0
            print("[RSLTXVICLoRAGuider] No reference_downscale_factor in metadata, using 1.0")

        print(f"[RSLTXVICLoRAGuider] Loaded: {lora_name} (downscale_factor={latent_downscale_factor})")

        # Apply LoRA
        if lora_strength != 0:
            m, _ = comfy.sd.load_lora_for_models(m, None, lora, lora_strength, 0)
            print(f"[RSLTXVICLoRAGuider] Applied LoRA (strength={lora_strength})")

        # --- Attention override (sage pattern) ---
        attn_func = None
        if attention_mode != "default":
            if attention_mode == "sage":
                from comfy.ldm.modules.attention import attention_sage
                attn_func = attention_sage
            elif attention_mode == "auto":
                from comfy.ldm.modules.attention import SAGE_ATTENTION_IS_AVAILABLE, attention_sage
                if SAGE_ATTENTION_IS_AVAILABLE:
                    attn_func = attention_sage
        if attn_func is not None:
            m.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = (
                lambda func, *args, **kwargs: attn_func(*args, **kwargs)
            )

        # --- Dimensions ---
        # Snap to 64-aligned so latent dims are always divisible by downscale_factor (2)
        # Use 128 alignment when upscale is on (halving must also produce even latent dims)
        align = 128 if upscale else 64
        width = max(align, round(width / align) * align)
        height = max(align, round(height / align) * align)
        out_width, out_height = width, height
        if upscale:
            width = width // 2
            height = height // 2
            print(f"[RSLTXVICLoRAGuider] Upscale mode: working at {width}x{height}")

        scale_factors = vae.downscale_index_formula
        time_sf, height_sf, width_sf = scale_factors
        latent_h = height // height_sf
        latent_w = width // width_sf
        latent_t = ((num_frames - 1) // time_sf) + 1

        # --- Encode control images ---
        control_latents = []
        control_masks = []
        num_control_frames = 0
        virtual_latent_length = latent_t
        dsf = int(latent_downscale_factor)

        controls = []
        if first_control is not None:
            controls.append((first_control, 0, "first"))
        if last_control is not None:
            controls.append((last_control, -1, "last"))

        for img, frame_idx, label in controls:
            # CRF preprocessing (crf=0 passes through unchanged)
            processed_frames = []
            for i in range(img.shape[0]):
                processed_frames.append(ltxv_preprocess(img[i], crf))
            processed = torch.stack(processed_frames)

            # Crop to valid frame count for LTXV temporal stride
            num_keep = ((processed.shape[0] - 1) // time_sf) * time_sf + 1
            processed = processed[:num_keep]

            # Resize to target / downscale_factor and VAE encode
            target_w_enc = int(latent_w * width_sf / latent_downscale_factor)
            target_h_enc = int(latent_h * height_sf / latent_downscale_factor)
            pixels = comfy.utils.common_upscale(
                processed.movedim(-1, 1),
                target_w_enc, target_h_enc,
                "bilinear", crop="disabled",
            ).movedim(1, -1)

            guide_latent = vae.encode(pixels[:, :, :, :3])
            print(f"[RSLTXVICLoRAGuider] {label}: {img.shape[2]}x{img.shape[1]} → "
                  f"encoded at {target_w_enc}x{target_h_enc}, latent {list(guide_latent.shape)}")

            # Dilate if downscale factor > 1 (sparse expansion + guide mask)
            guide_mask = None
            if dsf > 1:
                if latent_w % dsf != 0 or latent_h % dsf != 0:
                    raise ValueError(
                        f"Latent spatial size {latent_w}x{latent_h} must be "
                        f"divisible by latent_downscale_factor {dsf}"
                    )

                # Sparse expansion: place values at grid positions
                dilated_shape = guide_latent.shape[:3] + (
                    guide_latent.shape[3] * dsf,
                    guide_latent.shape[4] * dsf,
                )
                dilated = torch.zeros(
                    dilated_shape, device=guide_latent.device, dtype=guide_latent.dtype,
                )
                dilated[..., ::dsf, ::dsf] = guide_latent

                # Guide mask: -1.0 = empty position, 1.0 = valid data
                guide_mask = torch.full(
                    (dilated.shape[0], 1, dilated.shape[2], dilated.shape[3], dilated.shape[4]),
                    -1.0, device=guide_latent.device, dtype=guide_latent.dtype,
                )
                guide_mask[..., ::dsf, ::dsf] = 1.0

                guide_latent = dilated
                print(f"[RSLTXVICLoRAGuider] Dilated {label}: {list(guide_latent.shape)}")

            # Resolve frame index
            frame_idx_actual, latent_idx = LTXVAddGuide.get_latent_index(
                positive, virtual_latent_length,
                len(processed), frame_idx, scale_factors,
            )

            # Stamp keyframe indices on conditioning (for positional encoding)
            positive = LTXVAddGuide.add_keyframe_index(
                positive, frame_idx_actual, guide_latent, scale_factors,
            )
            negative = LTXVAddGuide.add_keyframe_index(
                negative, frame_idx_actual, guide_latent, scale_factors,
            )

            # Pre-compute control noise mask
            if guide_mask is not None:
                ctrl_mask = guide_mask - control_strength
            else:
                ctrl_mask = torch.full(
                    (1, 1, guide_latent.shape[2], 1, 1),
                    1.0 - control_strength,
                    dtype=guide_latent.dtype,
                )

            # Store on CPU for injection at sample() time
            control_latents.append(guide_latent.cpu())
            control_masks.append(ctrl_mask.cpu())
            num_control_frames += guide_latent.shape[2]
            virtual_latent_length += guide_latent.shape[2]

            print(f"[RSLTXVICLoRAGuider] {label} control at frame_idx={frame_idx_actual}")

        # --- Stamp frame rate on conditioning ---
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # --- Create guider ---
        guider = ICLoRAMultimodalGuider(
            m, positive, negative,
            control_latents=control_latents,
            control_masks=control_masks,
            num_control_frames=num_control_frames,
            video_cfg=cfg,
            audio_cfg=audio_cfg,
            stg_scale=stg_scale,
            stg_blocks=[int(s.strip()) for s in stg_blocks.split(",")],
            rescale=rescale,
            modality_scale=modality_scale,
            video_cfg_end=cfg_end if cfg_end >= 0 else None,
            stg_scale_end=stg_end if stg_end >= 0 else None,
        )

        print(f"[RSLTXVICLoRAGuider] Guider ready ({num_control_frames} control frames)")
        return (guider, out_width, out_height)
