import json
import os
import re

import torch
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from .prompt_formatter import DEFAULT_SYSTEM_PROMPT

# Class-level cache: {weight_path: (patcher, model, sp, has_vision)}
_model_cache = {}

# Gemma3 image placeholder token ID
IMAGE_TOKEN_ID = 262144


def _load_gemma3(weight_path):
    """Load a Gemma3 model (with vision) from a text_encoder safetensors file."""
    if weight_path in _model_cache:
        return _model_cache[weight_path]

    print(f"[RS Prompt Formatter Local] Loading {os.path.basename(weight_path)}...", flush=True)

    from comfy.text_encoders.llama import Gemma3_12B, Gemma3_12B_Config
    import sentencepiece

    sd, metadata = comfy.utils.load_torch_file(weight_path, safe_load=True, return_metadata=True)
    sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)

    # Extract SentencePiece tokenizer
    spiece_data = sd.pop("spiece_model", None)
    if spiece_data is None:
        raise RuntimeError(
            "[RS Prompt Formatter Local] No spiece_model found in weight file. "
            "This node requires a Gemma3 text encoder (e.g. from LTXV)."
        )
    if torch.is_tensor(spiece_data):
        spiece_data = spiece_data.numpy().tobytes()
    sp = sentencepiece.SentencePieceProcessor(model_proto=spiece_data)

    # Detect dtype and quantization
    dtype = sd.get("model.norm.weight", next(v for k, v in sd.items() if k.startswith("model."))).dtype
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant is not None:
        ops = comfy.ops.mixed_precision_ops(quant, dtype, full_precision_mm=True)
    else:
        ops = comfy.ops.manual_cast

    # Build full Gemma3_12B (LLM + vision model + projector)
    model = Gemma3_12B({}, dtype=dtype, device="cpu", operations=ops)

    # Load all weights — model.*, vision_model.*, multi_modal_projector.*
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        real_missing = [k for k in missing if "lm_head" not in k]
        if real_missing:
            print(f"[RS Prompt Formatter Local] Warning: missing keys: {real_missing[:5]}")

    has_vision = not any("vision_model" in k for k in missing)
    if has_vision:
        print("[RS Prompt Formatter Local] Vision model loaded — image input supported")
    else:
        print("[RS Prompt Formatter Local] No vision weights — image input will be ignored")

    # Wrap in ModelPatcher for GPU management
    load_device = mm.text_encoder_device()
    offload_device = mm.text_encoder_offload_device()
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

    parameters = comfy.utils.calculate_parameters(sd)
    print(f"[RS Prompt Formatter Local] Loaded Gemma3 ({parameters / 1e9:.1f}B params, {dtype})", flush=True)

    _model_cache[weight_path] = (patcher, model, sp, has_vision)
    return patcher, model, sp, has_vision


class RSPromptFormatterLocal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
            },
            "optional": {
                "first_image": ("IMAGE", {"tooltip": "First frame / opening image for the scene."}),
                "middle_image": ("IMAGE", {"tooltip": "Middle frame / key moment of the scene."}),
                "last_image": ("IMAGE", {"tooltip": "Last frame / ending image for the scene."}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Penalizes repeated tokens to encourage more detailed output. 1.0 = off, 1.1 = Ollama default."}),
                "repeat_window": ("INT", {"default": 64, "min": 0, "max": 512,
                                          "tooltip": "How many recent tokens to apply repeat penalty to. 0 = all generated tokens."}),
                "cache_file": ("STRING", {"default": "formatted_prompt.json",
                                          "tooltip": "JSON cache file. Re-runs generation only when input prompt changes."}),
                "output_dir": ("STRING", {"default": "",
                                          "tooltip": "Directory for cache file. Empty = ComfyUI output folder."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_prompt",)
    FUNCTION = "format_prompt"
    CATEGORY = "rs-nodes"

    def _resolve_cache_path(self, output_dir, cache_file):
        base = folder_paths.get_output_directory()
        if output_dir.strip():
            d = os.path.join(base, output_dir.strip())
        else:
            d = base
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, cache_file.strip())

    def _process_image(self, model, image, device):
        """Process a reference image through Gemma3's vision pipeline. Returns image embeddings."""
        import comfy.clip_model
        preprocessed = comfy.clip_model.clip_preprocess(
            image, size=model.image_size,
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop=True,
        )
        pixel_values = preprocessed.to(device, dtype=torch.float32)
        vision_out = model.vision_model(pixel_values)[0]
        image_embeds = model.multi_modal_projector(vision_out)
        return image_embeds  # [1, 256, hidden_size]

    def _build_embeds_with_image(self, model, input_ids, image_embeds_list, device):
        """Build embedding tensor, replacing image placeholder tokens with vision embeddings."""
        tokens_tensor = torch.LongTensor([input_ids]).to(device)
        text_embeds = model.model.embed_tokens(tokens_tensor)  # [1, seq_len, hidden_size]

        # Find image placeholder positions
        placeholder_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_TOKEN_ID]

        if not placeholder_positions or not image_embeds_list:
            return text_embeds

        # The projector weights were trained against HuggingFace's flow where text
        # embeddings are pre-scaled by sqrt(hidden_size). ComfyUI's Llama2_.forward()
        # applies that scaling later to the entire embeds tensor. To prevent double-
        # scaling the image tokens, divide them by the scale factor here.
        scale = model.model.config.hidden_size ** 0.5

        # Replace each placeholder with its corresponding image embeddings.
        # Process in reverse order so earlier positions don't shift.
        for pos, img_embeds in reversed(list(zip(placeholder_positions, image_embeds_list))):
            before = text_embeds[:, :pos, :]
            after = text_embeds[:, pos + 1:, :]
            img = img_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype) / scale
            text_embeds = torch.cat([before, img, after], dim=1)

        return text_embeds

    def _generate(self, patcher, model, sp, prompt_text, system_prompt,
                  max_tokens, temperature, top_k, repeat_penalty, repeat_window,
                  images=None, has_vision=False):
        """Run autoregressive generation using the Gemma3 model."""
        mm.load_models_gpu([patcher])
        device = mm.get_torch_device()

        # images is a list of (label, tensor) pairs
        image_embeds_list = []

        if images and has_vision:
            print(f"[RS Prompt Formatter Local] Processing {len(images)} reference image(s)...", flush=True)
            # Build chat template with labeled image placeholders
            parts = [f"<start_of_turn>user\n{system_prompt}\n\nReference images:\n"]
            token_parts = [sp.encode(parts[0])]

            for label, img_tensor in images:
                with torch.no_grad():
                    img_embeds = self._process_image(model, img_tensor, device)
                image_embeds_list.append(img_embeds)
                # Add label text, then placeholder, then newline
                label_text = f"{label}: "
                token_parts.append(sp.encode(label_text))
                token_parts.append([IMAGE_TOKEN_ID])
                token_parts.append(sp.encode("\n"))

            after_text = f"\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
            token_parts.append(sp.encode(after_text))

            input_ids = []
            for part in token_parts:
                input_ids.extend(part)
        else:
            chat = (
                f"<start_of_turn>user\n{system_prompt}\n\n{prompt_text}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            input_ids = sp.encode(chat)

        print("[RS Prompt Formatter Local] Generating:", flush=True)

        # Build initial embeddings
        with torch.no_grad():
            if image_embeds_list:
                embeds = self._build_embeds_with_image(model, input_ids, image_embeds_list, device)
            else:
                tokens_tensor = torch.LongTensor([input_ids]).to(device)
                embeds = model.model.embed_tokens(tokens_tensor)

            # Use the built-in generate method (has KV cache, proper logits, dtype handling)
            eos_id = sp.eos_id()
            eot_id = sp.piece_to_id("<end_of_turn>")
            stop_tokens = [eos_id]
            if eot_id is not None:
                stop_tokens.append(eot_id)

            generated_ids = model.generate(
                embeds=embeds,
                do_sample=temperature > 0,
                max_length=max_tokens,
                temperature=max(temperature, 1e-6),
                top_k=top_k,
                repetition_penalty=repeat_penalty,
                stop_tokens=stop_tokens,
                initial_tokens=input_ids,
            )

        output = sp.decode(generated_ids)

        # Strip <think>...</think> blocks (including partial/unclosed)
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        output = re.sub(r"<think>.*", "", output, flags=re.DOTALL)
        output = output.replace("<end_of_turn>", "")

        result = output.strip()
        print(f"\n[RS Prompt Formatter Local] Output ({len(generated_ids)} tokens):\n{result}", flush=True)
        return result

    def format_prompt(self, text_encoder, prompt, system_prompt,
                      first_image=None, middle_image=None, last_image=None,
                      max_tokens=1024, temperature=0.8, top_k=40,
                      repeat_penalty=1.1, repeat_window=64,
                      cache_file="formatted_prompt.json", output_dir=""):
        cache_path = self._resolve_cache_path(output_dir, cache_file)

        # Build labeled image list from connected inputs
        images = []
        if first_image is not None:
            images.append(("First frame", first_image))
        if middle_image is not None:
            images.append(("Middle frame", middle_image))
        if last_image is not None:
            images.append(("Last frame", last_image))

        # Check JSON cache — skip generation if prompt hasn't changed and no images
        if not images and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if cache.get("prompt") == prompt and cache.get("system_prompt") == system_prompt:
                    print("[RS Prompt Formatter Local] Prompt unchanged — using cached output")
                    return (cache["output"],)
            except (json.JSONDecodeError, KeyError):
                pass

        weight_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        patcher, model, sp, has_vision = _load_gemma3(weight_path)

        if images and not has_vision:
            print("[RS Prompt Formatter Local] Warning: images provided but model has no vision weights — ignoring images")
            images = []

        formatted = self._generate(patcher, model, sp, prompt, system_prompt,
                                   max_tokens, temperature, top_k,
                                   repeat_penalty, repeat_window,
                                   images=images or None,
                                   has_vision=has_vision)

        if not formatted:
            raise RuntimeError("[RS Prompt Formatter Local] Model returned empty output")

        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "system_prompt": system_prompt, "output": formatted}, f, indent=2)
        print(f"[RS Prompt Formatter Local] Saved output to {cache_path}")

        return (formatted,)
