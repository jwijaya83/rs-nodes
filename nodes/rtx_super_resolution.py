"""RTX Video Super Resolution node wrapping NVIDIA nvvfx.VideoSuperRes.

Exposes all QualityLevel modes: standard upscaling, denoise, deblur, and
highbitrate upscaling.  Denoise and deblur modes output the same resolution
as the input; upscale modes apply the user-chosen scale or target dimensions.
"""

import logging
import math
from enum import Enum
from typing import TypedDict

import torch
import comfy.utils

from comfy_api.latest import io

logger = logging.getLogger(__name__)

try:
    import nvvfx
    from nvvfx.effects import QualityLevel
    _NVVFX_AVAILABLE = True
except ImportError:
    _NVVFX_AVAILABLE = False
    QualityLevel = None
    logger.warning(
        "nvvfx not found — RSRTXSuperResolution will be unavailable. "
        "Install the NVIDIA Video Effects SDK Python bindings to enable it."
    )


class ResizeType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class Mode(str, Enum):
    UPSCALE = "upscale"
    UPSCALE_HIGHBITRATE = "upscale (high bitrate)"
    DENOISE = "denoise"
    DEBLUR = "deblur"


# Map (mode, quality) -> QualityLevel enum value
_QUALITY_MAP = {}
if _NVVFX_AVAILABLE:
    _QUALITY_MAP = {
        (Mode.UPSCALE, "LOW"): QualityLevel.LOW,
        (Mode.UPSCALE, "MEDIUM"): QualityLevel.MEDIUM,
        (Mode.UPSCALE, "HIGH"): QualityLevel.HIGH,
        (Mode.UPSCALE, "ULTRA"): QualityLevel.ULTRA,
        (Mode.UPSCALE_HIGHBITRATE, "LOW"): QualityLevel.HIGHBITRATE_LOW,
        (Mode.UPSCALE_HIGHBITRATE, "MEDIUM"): QualityLevel.HIGHBITRATE_MEDIUM,
        (Mode.UPSCALE_HIGHBITRATE, "HIGH"): QualityLevel.HIGHBITRATE_HIGH,
        (Mode.UPSCALE_HIGHBITRATE, "ULTRA"): QualityLevel.HIGHBITRATE_ULTRA,
        (Mode.DENOISE, "LOW"): QualityLevel.DENOISE_LOW,
        (Mode.DENOISE, "MEDIUM"): QualityLevel.DENOISE_MEDIUM,
        (Mode.DENOISE, "HIGH"): QualityLevel.DENOISE_HIGH,
        (Mode.DENOISE, "ULTRA"): QualityLevel.DENOISE_ULTRA,
        (Mode.DEBLUR, "LOW"): QualityLevel.DEBLUR_LOW,
        (Mode.DEBLUR, "MEDIUM"): QualityLevel.DEBLUR_MEDIUM,
        (Mode.DEBLUR, "HIGH"): QualityLevel.DEBLUR_HIGH,
        (Mode.DEBLUR, "ULTRA"): QualityLevel.DEBLUR_ULTRA,
    }

_SAME_RES_MODES = {Mode.DENOISE, Mode.DEBLUR}
_MAX_PIXELS = 1024 * 1024 * 16


def _align8(value: int) -> int:
    return max(8, round(value / 8) * 8)


class RSRTXSuperResolution(io.ComfyNode):
    """RTX Video Super Resolution — upscale, denoise, or deblur via nvvfx."""

    class ResizeTypedDict(TypedDict):
        resize_type: ResizeType
        scale: float
        width: int
        height: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RSRTXSuperResolution",
            display_name="RS RTX Super Resolution",
            category="rs-nodes",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "denoise", "deblur"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "mode",
                    options=[m.value for m in Mode],
                    default=Mode.UPSCALE.value,
                    tooltip="upscale: standard AI upscaling. "
                            "upscale (high bitrate): preserves detail from clean sources. "
                            "denoise: same-res noise/artifact removal. "
                            "deblur: same-res sharpening.",
                ),
                io.Combo.Input(
                    "quality",
                    options=["LOW", "MEDIUM", "HIGH", "ULTRA"],
                    default="ULTRA",
                ),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions. "
                            "Ignored for denoise/deblur modes (same resolution).",
                    options=[
                        io.DynamicCombo.Option(ResizeType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01,
                                           tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=3840, min=64, max=8192, step=8,
                                         tooltip="Target width in pixels."),
                            io.Int.Input("height", default=2160, min=64, max=8192, step=8,
                                         tooltip="Target height in pixels."),
                        ]),
                    ],
                ),
            ],
            outputs=[
                io.Image.Output("images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, mode: str, quality: str,
                resize_type: "RSRTXSuperResolution.ResizeTypedDict") -> io.NodeOutput:
        if not _NVVFX_AVAILABLE:
            raise RuntimeError(
                "nvvfx is not installed. "
                "Install the NVIDIA Video Effects SDK Python bindings to use this node."
            )

        B, H, W, C = images.shape

        # Determine output resolution
        if mode in {m.value for m in _SAME_RES_MODES}:
            out_w = _align8(W)
            out_h = _align8(H)
        else:
            selected_type = resize_type["resize_type"]
            if selected_type == ResizeType.SCALE_BY:
                out_w = _align8(int(W * resize_type["scale"]))
                out_h = _align8(int(H * resize_type["scale"]))
            elif selected_type == ResizeType.TARGET_DIMENSIONS:
                out_w = _align8(resize_type["width"])
                out_h = _align8(resize_type["height"])
            else:
                raise ValueError(f"Unsupported resize type: {selected_type}")

        mode_enum = Mode(mode)
        quality_level = _QUALITY_MAP[(mode_enum, quality)]

        # Detailed debug output
        logger.info("=" * 60)
        logger.info("RSRTXSuperResolution")
        logger.info("  Mode:       %s", mode)
        logger.info("  Quality:    %s (nvvfx=%s)", quality, quality_level)
        logger.info("  Input:      %dx%d (%d frames, %s)", W, H, B, images.dtype)
        logger.info("  Output:     %dx%d (%.1fMP)", out_w, out_h, out_w * out_h / 1e6)
        if mode_enum not in _SAME_RES_MODES:
            selected_type = resize_type["resize_type"]
            if selected_type == ResizeType.SCALE_BY:
                logger.info("  Resize:     scale %.2fx", resize_type["scale"])
            else:
                logger.info("  Resize:     target %dx%d", resize_type["width"], resize_type["height"])
        else:
            logger.info("  Resize:     same resolution (ignored for %s)", mode)
        logger.info("=" * 60)

        out_pixels = out_w * out_h
        max_batch = max(1, _MAX_PIXELS // out_pixels)

        output_frames = []
        pbar = comfy.utils.ProgressBar(B)

        with nvvfx.VideoSuperRes(quality_level) as sr:
            sr.output_width = out_w
            sr.output_height = out_h
            sr.load()
            logger.info("nvvfx model loaded, processing %d frames...", B)

            import time
            t_start = time.perf_counter()

            for i in range(B):
                frame_cuda = (
                    images[i]
                    .permute(2, 0, 1)
                    .contiguous()
                    .cuda()
                )

                result = sr.run(frame_cuda)
                out_tensor = torch.from_dlpack(result.image).clone()

                output_frames.append(out_tensor.permute(1, 2, 0).cpu())
                pbar.update(1)

            elapsed = time.perf_counter() - t_start
            fps = B / elapsed if elapsed > 0 else 0
            logger.info("Processed %d frames in %.1fs (%.1f fps)", B, elapsed, fps)

        output = torch.stack(output_frames)
        return io.NodeOutput(output)


