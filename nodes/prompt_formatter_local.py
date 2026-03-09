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

# Class-level cache: {weight_path: (patcher, llm, tokenizer)}
_model_cache = {}


def _load_gemma3(weight_path):
    """Load a Gemma3 model from a text_encoder safetensors file. Returns (patcher, llm, sp)."""
    if weight_path in _model_cache:
        return _model_cache[weight_path]

    print(f"[RS Prompt Formatter Local] Loading {os.path.basename(weight_path)}...", flush=True)

    from comfy.text_encoders.llama import Llama2_, Gemma3_12B_Config
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

    # Build LLM (Llama2_ only — skip vision model / projector)
    config = Gemma3_12B_Config()
    llm = Llama2_(config, device="cpu", dtype=dtype, ops=ops)

    # Strip "model." prefix so keys match Llama2_ structure
    llm_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            llm_sd[k[len("model."):]] = v
    missing, unexpected = llm.load_state_dict(llm_sd, strict=False)
    if missing:
        # Filter out expected missing keys (lm_head if present)
        real_missing = [k for k in missing if "lm_head" not in k]
        if real_missing:
            print(f"[RS Prompt Formatter Local] Warning: missing keys: {real_missing[:5]}")

    # Wrap in ModelPatcher for GPU management
    load_device = mm.text_encoder_device()
    offload_device = mm.text_encoder_offload_device()
    patcher = comfy.model_patcher.ModelPatcher(llm, load_device=load_device, offload_device=offload_device)

    parameters = comfy.utils.calculate_parameters(llm_sd)
    print(f"[RS Prompt Formatter Local] Loaded Gemma3 ({parameters / 1e9:.1f}B params, {dtype})", flush=True)

    _model_cache[weight_path] = (patcher, llm, sp)
    return patcher, llm, sp


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
                "reference_image": ("IMAGE",),
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

    def _generate(self, patcher, llm, sp, prompt_text, system_prompt,
                  max_tokens, temperature, top_k, repeat_penalty, repeat_window):
        """Run autoregressive generation using the Gemma3 model."""
        mm.load_models_gpu([patcher])
        device = mm.get_torch_device()

        # Gemma3 chat template
        chat = (
            f"<start_of_turn>user\n{system_prompt}\n\n{prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        input_ids = sp.encode(chat)

        # Special token IDs
        eos_id = sp.eos_id()
        eot_id = sp.piece_to_id("<end_of_turn>")

        prompt_len = len(input_ids)
        print("[RS Prompt Formatter Local] Generating:", flush=True)

        for _ in range(max_tokens):
            tokens = torch.LongTensor([input_ids]).to(device)

            with torch.no_grad():
                hidden, _ = llm(tokens)

            last_hidden = hidden[0, -1, :]  # [hidden_size]

            # Weight-tied lm_head: logits = hidden @ embed_weight^T
            embed_weight = llm.embed_tokens.weight
            logits = (last_hidden @ embed_weight.T).float()  # [vocab_size]

            # Repeat penalty — penalize tokens already generated
            if repeat_penalty > 1.0:
                generated = input_ids[prompt_len:]
                if repeat_window > 0:
                    generated = generated[-repeat_window:]
                if generated:
                    penalty_ids = torch.LongTensor(generated).to(logits.device)
                    penalty_logits = logits[penalty_ids]
                    # Divide positive logits, multiply negative logits
                    logits[penalty_ids] = torch.where(
                        penalty_logits > 0,
                        penalty_logits / repeat_penalty,
                        penalty_logits * repeat_penalty,
                    )

            # Temperature + top-k sampling
            if temperature > 0:
                logits = logits / temperature
                top_k_logits, top_k_ids = torch.topk(logits, min(top_k, logits.shape[0]))
                probs = torch.softmax(top_k_logits, dim=-1)
                next_id = top_k_ids[torch.multinomial(probs, 1)].item()
            else:
                next_id = logits.argmax().item()

            input_ids.append(next_id)

            token_text = sp.decode([next_id])
            print(token_text, end="", flush=True)

            if next_id == eos_id:
                break
            if eot_id is not None and next_id == eot_id:
                break

        print(flush=True)

        # Decode generated portion only
        output = sp.decode(input_ids[prompt_len:])

        # Strip <think>...</think> blocks (including partial/unclosed)
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        output = re.sub(r"<think>.*", "", output, flags=re.DOTALL)
        output = output.replace("<end_of_turn>", "")

        return output.strip()

    def format_prompt(self, text_encoder, prompt, system_prompt,
                      reference_image=None, max_tokens=1024, temperature=0.8, top_k=40,
                      repeat_penalty=1.1, repeat_window=64,
                      cache_file="formatted_prompt.json", output_dir=""):
        cache_path = self._resolve_cache_path(output_dir, cache_file)

        # Check JSON cache — skip generation if prompt hasn't changed
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if cache.get("prompt") == prompt and cache.get("system_prompt") == system_prompt:
                    print("[RS Prompt Formatter Local] Prompt unchanged — using cached output")
                    return (cache["output"],)
            except (json.JSONDecodeError, KeyError):
                pass

        weight_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        patcher, llm, sp = _load_gemma3(weight_path)

        formatted = self._generate(patcher, llm, sp, prompt, system_prompt,
                                   max_tokens, temperature, top_k,
                                   repeat_penalty, repeat_window)

        if not formatted:
            raise RuntimeError("[RS Prompt Formatter Local] Model returned empty output")

        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "system_prompt": system_prompt, "output": formatted}, f, indent=2)
        print(f"[RS Prompt Formatter Local] Saved output to {cache_path}")

        return (formatted,)
