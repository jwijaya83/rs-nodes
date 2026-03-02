"""TTM (Time-to-Move) guider for LTXV video generation.

Implements dual-clock denoising for motion-controlled video generation.
After each denoising step, replaces masked latent regions with noisy reference
latents to enforce motion guidance while allowing free denoising elsewhere.

Reference: "Time-to-Move: Training-Free Motion Controlled Video Generation
via Dual-Clock Denoising" (arXiv:2511.08633)
"""

import math

import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.model_sampling
import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
import comfy.hooks
import comfy.patcher_extension
import latent_preview
import node_helpers
from comfy.samplers import (process_conds, cast_to_load_options,
                            preprocess_conds_hooks, get_total_hook_groups_in_conds,
                            filter_registered_hooks_on_conds)

from ..utils.multimodal_guider import STGFlag, STGBlockWrapper, MultimodalGuider


# ---------------------------------------------------------------------------
# TTM Guider — runs its own sampling loop with post-step mask replacement
# ---------------------------------------------------------------------------

class LTXVTTMGuider(comfy.samplers.CFGGuider):
    """CFG guider with TTM dual-clock denoising for motion control.

    Runs its own euler sampling loop. After each denoising step:
      - If within the TTM window: replace masked region of the latent
        with a flow-matching noised version of the reference
      - Outside the window: let the model denoise freely
    """

    def __init__(
        self,
        model,
        positive,
        negative,
        reference_latents,
        mask,
        cfg=3.0,
        cfg_end=None,
        ttm_strength=0.5,
        rescale=0.7,
        max_shift=2.05,
        base_shift=0.95,
        stg_scale=0.0,
        stg_scale_end=None,
        stg_blocks=None,
    ):
        super().__init__(model)
        self.cfg_value = cfg
        self.cfg_end = cfg_end if cfg_end is not None else cfg
        self.rescale_val = rescale
        self.reference_latents = reference_latents
        self.ttm_mask = mask
        self.ttm_strength = ttm_strength
        self.max_shift = max_shift
        self.base_shift = base_shift
        self._current_step = 0
        self._total_steps = 1

        # STG setup
        self.stg_scale = stg_scale
        self.stg_scale_end = stg_scale_end if stg_scale_end is not None else stg_scale
        self.stg_blocks = stg_blocks if stg_blocks is not None else [29]
        self.stg_flag = STGFlag(skip_layers=list(self.stg_blocks))
        self._need_stg = stg_scale > 0 or (stg_scale_end is not None and stg_scale_end > 0)
        if self._need_stg:
            MultimodalGuider._patch_model(self.model_patcher, self.stg_flag)

        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """CFG + STG noise prediction with per-step interpolation."""
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        step = self._current_step
        self._current_step += 1

        # Per-step interpolation
        t = step / max(self._total_steps - 1, 1)
        cur_cfg = float(self.cfg_value + t * (self.cfg_end - self.cfg_value))
        cur_stg = float(self.stg_scale + t * (self.stg_scale_end - self.stg_scale))

        # Pass 1: positive
        pos_out = comfy.samplers.calc_cond_batch(
            model, [positive], x, timestep, model_options
        )[0]

        # Pass 2: negative (CFG)
        need_cfg = cur_cfg != 1.0
        if need_cfg and negative is not None:
            neg_out = comfy.samplers.calc_cond_batch(
                model, [negative], x, timestep, model_options
            )[0]
        else:
            neg_out = pos_out

        # Pass 3: STG (perturbed)
        stg_out = pos_out
        if self._need_stg and cur_stg > 0:
            to = model_options.setdefault("transformer_options", {})
            try:
                to["ptb_index"] = 0
                to["stg_indexes"] = [0]  # video self-attention
                self.stg_flag.do_skip = True
                stg_out = comfy.samplers.calc_cond_batch(
                    model, [positive], x, timestep, model_options
                )[0]
            finally:
                self.stg_flag.do_skip = False
                to.pop("ptb_index", None)
                to.pop("stg_indexes", None)

        # Guidance formula: pos + (cfg-1)*(pos-neg) + stg*(pos-perturbed)
        noise_pred = (
            pos_out
            + (cur_cfg - 1) * (pos_out - neg_out)
            + cur_stg * (pos_out - stg_out)
        )

        if self.rescale_val != 0:
            factor = pos_out.std() / noise_pred.std()
            factor = self.rescale_val * factor + (1 - self.rescale_val)
            noise_pred = noise_pred * factor

        return noise_pred

    def inner_sample(self, noise, latent_image, device, sampler, sigmas,
                     denoise_mask, callback, disable_pbar, seed,
                     latent_shapes=None):
        """Custom sampling loop with post-step TTM mask replacement."""
        if latent_image is not None and torch.count_nonzero(latent_image) > 0:
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(
            self.inner_model, noise, self.conds, device,
            latent_image, denoise_mask, seed, latent_shapes=latent_shapes,
        )

        total_steps = sigmas.shape[-1] - 1
        self._current_step = 0
        self._total_steps = total_steps
        ttm_end_step = max(1, int(total_steps * self.ttm_strength))

        # Resize reference and mask to match generation shape
        ref = self.reference_latents.to(device)
        mask = self.ttm_mask.to(device)
        target_shape = noise.shape[2:]
        if ref.shape[2:] != target_shape:
            print(f"[TTMGuider] WARNING: Resizing reference {list(ref.shape[2:])} → {list(target_shape)}")
            print(f"[TTMGuider] For best quality, set TTM guider width/height to match generation resolution")
            ref = F.interpolate(ref.float(), size=target_shape, mode='trilinear', align_corners=False)
            mask = F.interpolate(mask.float(), size=target_shape, mode='nearest')
            mask = (mask > 0.5).float()

        # Diagnostics
        mask_coverage = mask.sum().item() / max(mask.numel(), 1) * 100
        print(f"[TTMGuider] Mask coverage: {mask_coverage:.1f}% active")
        print(f"[TTMGuider] Reference latent stats: mean={ref.mean():.4f}, std={ref.std():.4f}")

        # Recompute shift from actual generation dimensions
        tokens = math.prod(target_shape)
        x1, x2 = 1024, 4096
        mm_shift = (self.max_shift - self.base_shift) / (x2 - x1)
        b = self.base_shift - mm_shift * x1
        shift = tokens * mm_shift + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling_obj = ModelSamplingAdvanced(self.model_patcher.model.model_config)
        model_sampling_obj.set_parameters(shift=shift)
        self.model_patcher.add_object_patch("model_sampling", model_sampling_obj)

        print(f"[TTMGuider] Generation: {list(target_shape)}, shift={shift:.3f}")
        print(f"[TTMGuider] TTM active for steps 0-{ttm_end_step} of {total_steps}")

        # Build model callable for the sampling loop
        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas

        s_in = noise.new_ones([noise.shape[0]])
        base_noise = noise.clone()

        # Initialize x from noisy reference (not pure noise) for structural info
        sigma_start = sigmas[0]
        noisy_ref_init = (1 - sigma_start) * ref + sigma_start * noise
        if denoise_mask is not None:
            # Preserve first_image/last_image frames, use noisy reference elsewhere
            x = latent_image * (1 - denoise_mask) + noisy_ref_init * denoise_mask
        else:
            x = noisy_ref_init

        # Euler RF sampling loop with post-step TTM replacement
        try:
            for i in range(total_steps):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]

                # Blend preserved frames into x before model sees it
                # (matches KSamplerX0Inpaint pre-model blending for LTXV)
                if denoise_mask is not None:
                    x = x * denoise_mask + latent_image * (1 - denoise_mask)

                # Model prediction (goes through our predict_noise → CFG + STG)
                denoised = self.outer_predict_noise(
                    x, sigma * s_in, extra_model_options, seed
                )

                # Apply denoise mask on denoised output (matches KSamplerX0Inpaint)
                if denoise_mask is not None:
                    denoised = denoised * denoise_mask + latent_image * (1 - denoise_mask)

                # Euler step (RF flow matching)
                if sigma_next > 0:
                    x = denoised + (sigma_next / sigma) * (x - denoised)
                else:
                    x = denoised

                # TTM: replace masked region AFTER the step
                if i < ttm_end_step and sigma_next > 0:
                    sigma_val = sigma_next.view((1,) * len(ref.shape)).to(x)
                    noisy_ref = (1 - sigma_val) * ref + sigma_val * base_noise
                    x = x * (1 - mask) + noisy_ref.to(x) * mask

                # Callback for progress bar + latent preview
                if callback is not None:
                    callback(i, denoised, x, total_steps)

            return self.inner_model.process_latent_out(x.to(torch.float32))
        finally:
            # Always free GPU tensors, even on error/interrupt
            del ref, mask, base_noise
            self.reference_latents = None
            self.ttm_mask = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RSLTXVTTMGuider:
    """Creates a TTM (Time-to-Move) guider for motion-controlled LTXV generation.

    Takes a reference video and mask, encodes them, and produces a GUIDER
    that enforces reference motion during sampling. Plug the output into
    RSLTXVGenerate's guider input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("MODEL",),
                "positive":        ("CONDITIONING",),
                "negative":        ("CONDITIONING",),
                "vae":             ("VAE",),
                "reference_video": ("IMAGE",),
                "mask":            ("IMAGE",),
                "width":           ("INT",   {"default": 768,  "min": 64,  "max": 8192, "step": 32}),
                "height":          ("INT",   {"default": 512,  "min": 64,  "max": 8192, "step": 32}),
            },
            "optional": {
                "cfg":             ("FLOAT", {"default": 3.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end":         ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "ttm_strength":    ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,   "step": 0.05}),
                "rescale":         ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0,   "step": 0.01}),
                "stg_scale":       ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 10.0,  "step": 0.1}),
                "stg_end":         ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0,  "step": 0.1}),
                "stg_blocks":      ("STRING", {"default": "29"}),
                "frame_rate":      ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "max_shift":       ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift":      ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "upscale":         ("BOOLEAN", {"default": False}),
                "attention_mode":  (["auto", "default", "sage"],),
            },
        }

    RETURN_TYPES = ("GUIDER", "INT", "INT")
    RETURN_NAMES = ("guider", "width", "height")
    FUNCTION = "create_guider"
    CATEGORY = "rs-nodes"

    def create_guider(
        self, model, positive, negative, vae, reference_video, mask,
        width=768, height=512,
        cfg=3.0, cfg_end=-1.0, ttm_strength=0.5, rescale=0.7,
        stg_scale=0.0, stg_end=-1.0, stg_blocks="29",
        frame_rate=25.0, max_shift=2.05, base_shift=0.95,
        upscale=False, attention_mode="auto",
    ):
        m = model.clone()

        # Attention override
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

        # Match generate node: halve resolution when upscale is enabled
        if upscale:
            width = width // 2
            height = height // 2
            print(f"[RSLTXVTTMGuider] Upscale mode: encoding reference at {width}x{height}")

        # Compute target resolution preserving reference aspect ratio
        src_h, src_w = reference_video.shape[1], reference_video.shape[2]
        aspect = src_w / src_h
        total_pixels = width * height
        target_h = int(math.sqrt(total_pixels / aspect))
        target_w = int(target_h * aspect)
        target_w = max(64, round(target_w / 32) * 32)
        target_h = max(64, round(target_h / 32) * 32)

        if src_h != target_h or src_w != target_w:
            print(f"[RSLTXVTTMGuider] Resizing to match aspect ratio: {src_w}x{src_h} → {target_w}x{target_h}")
        else:
            print(f"[RSLTXVTTMGuider] Reference resolution: {target_w}x{target_h}")

        reference_video = comfy.utils.common_upscale(
            reference_video.movedim(-1, 1), target_w, target_h, "bilinear", "center"
        ).movedim(1, -1)

        print(f"[RSLTXVTTMGuider] Mask input: shape={list(mask.shape)}, dim={mask.dim()}, "
              f"min={mask.min():.3f}, max={mask.max():.3f}")

        # Normalize mask to [B, H, W] regardless of input format
        if mask.dim() == 4:
            # IMAGE format [B, H, W, C] → grayscale
            mask = mask[:, :, :, :3].mean(dim=-1)
        elif mask.dim() == 2:
            # Single frame [H, W] → add batch
            mask = mask.unsqueeze(0)
        # Now mask is [B, H, W]

        # Spatially resize mask to match reference
        mask = comfy.utils.common_upscale(
            mask.unsqueeze(1), target_w, target_h, "bilinear", "center"
        ).squeeze(1)  # [B, target_h, target_w]

        print(f"[RSLTXVTTMGuider] Mask after resize: {mask.shape[0]} frames, "
              f"per-frame coverage: {['%.1f%%' % (f.mean().item() * 100) for f in mask[:4]]}...")

        # Pad frames to LTX VAE requirement (1 + 8*x frames)
        num_frames = reference_video.shape[0]
        target_frames = ((num_frames - 1 + 7) // 8) * 8 + 1
        if num_frames != target_frames:
            pad_count = target_frames - num_frames
            padding = reference_video[-1:].expand(pad_count, -1, -1, -1)
            reference_video = torch.cat([reference_video, padding], dim=0)
            # Pad mask to match
            mask_pad = mask[-1:].expand(pad_count, -1, -1)
            mask = torch.cat([mask, mask_pad], dim=0)
            print(f"[RSLTXVTTMGuider] Padded reference+mask: {num_frames} → {target_frames} frames")

        print(f"[RSLTXVTTMGuider] Encoding reference video ({reference_video.shape[0]} frames at {target_w}x{target_h})")
        reference_latents = vae.encode(reference_video[:, :, :, :3])
        _, _, lat_t, lat_h, lat_w = reference_latents.shape
        print(f"[RSLTXVTTMGuider] Reference latents: {reference_latents.shape}")

        # Stamp frame rate onto conditioning
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # Convert [B, H, W] mask to [1, 1, T, H, W] latent-space mask
        mask_5d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, B, H, W]
        mask_5d = F.interpolate(mask_5d.float(), size=(lat_t, lat_h, lat_w), mode='nearest')
        mask_5d = (mask_5d > 0.5).float()

        total_coverage = mask_5d.mean().item() * 100
        print(f"[RSLTXVTTMGuider] Mask coverage: {total_coverage:.1f}% (mask=1 → replace with reference)")
        print(f"[RSLTXVTTMGuider] TTM strength: {ttm_strength}")

        # Store on CPU — moved to GPU on demand in inner_sample
        reference_latents = reference_latents.cpu()
        mask_5d = mask_5d.cpu()

        guider = LTXVTTMGuider(
            m, positive, negative,
            reference_latents=reference_latents,
            mask=mask_5d,
            cfg=cfg,
            cfg_end=cfg_end if cfg_end >= 0 else None,
            ttm_strength=ttm_strength,
            rescale=rescale,
            max_shift=max_shift,
            base_shift=base_shift,
            stg_scale=stg_scale,
            stg_scale_end=stg_end if stg_end >= 0 else None,
            stg_blocks=[int(s.strip()) for s in stg_blocks.split(",")],
        )

        return (guider, target_w, target_h)
