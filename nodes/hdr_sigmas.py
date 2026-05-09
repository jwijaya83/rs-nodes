"""HDR sigmas node.

Outputs the official Lightricks LTX-2.3 HDR IC-LoRA manual sigma curve
straight from their reference workflow
(example_workflows/2.3/LTX-2.3_ICLoRA_HDR_Distilled.json):

    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0

The curve is hand-tuned, not shift-warped — five tiny steps near σ=1
that barely touch the latent (preserves the LogC3 / HDR scene linearity
the IC-LoRA was trained on), followed by four aggressive descents that
do most of the actual denoising.

Wire the output into RSLTXVGenerate.sigmas to bypass the node's
internal scheduler. Pair with single-pass mode (no re-diffusion) to
match the official workflow exactly.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# Official Lightricks HDR sigma curve (8 steps, 9 points including 0.0).
# Don't tweak unless you've validated against their reference output.
HDR_OFFICIAL_8 = (
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
)


class RSHDRSigmas:
    """LTX-2.3 HDR IC-LoRA manual sigma curve."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    ["official_8 (Lightricks reference)", "custom"],
                    {
                        "default": "official_8 (Lightricks reference)",
                        "tooltip": (
                            "official_8: the exact curve from Lightricks' "
                            "LTX-2.3_ICLoRA_HDR_Distilled.json reference "
                            "workflow. custom: parse the comma-separated "
                            "values from the custom_sigmas widget."
                        ),
                    },
                ),
            },
            "optional": {
                "custom_sigmas": (
                    "STRING",
                    {
                        "default": (
                            "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                            "0.909375, 0.725, 0.421875, 0.0"
                        ),
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated sigma values, descending, "
                            "ending in 0.0. Only used when preset=custom."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "build"
    CATEGORY = "rs-nodes"

    def build(self, preset: str, custom_sigmas: str = "") -> tuple[torch.Tensor]:
        if preset.startswith("official"):
            values = list(HDR_OFFICIAL_8)
        else:
            try:
                values = [float(x.strip()) for x in custom_sigmas.split(",") if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"custom_sigmas could not be parsed as comma-separated "
                    f"floats: {e}"
                )
            if len(values) < 2:
                raise ValueError(
                    "custom_sigmas needs at least 2 values (e.g. '1.0, 0.0')."
                )
            if values[-1] != 0.0:
                # Sampler expects the schedule to end at zero — append it
                # rather than fail, since this is a common forget-to-add.
                values.append(0.0)
                logger.warning(
                    "RSHDRSigmas: appended terminal 0.0 to custom_sigmas."
                )

        sigmas = torch.tensor(values, dtype=torch.float32)
        logger.info(f"RSHDRSigmas: {len(sigmas) - 1} steps, {sigmas.tolist()}")
        return (sigmas,)
