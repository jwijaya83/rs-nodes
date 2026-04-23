"""LogC3 decode + tone-map for LTX-2.3 HDR IC-LoRA output.

The HDR IC-LoRA emits frames encoded in ARRI LogC3 (EI 800) so the model's
[0, 1] output range can represent an extended-dynamic-range signal. This
node inverts that encoding to get linear-light HDR values, optionally
applies an exposure adjustment, and produces a Reinhard-tonemapped SDR
preview for display / video encoding.

Pipeline:
  IMAGE (LogC3, from VAE Decode) ─┬─> hdr_linear (IMAGE, linear HDR for EXR)
                                  └─> sdr_preview (IMAGE, tonemapped for MP4/ProRes)

Math transcribed verbatim from Lightricks' ComfyUI-LTXVideo hdr.py so
values match their reference implementation exactly.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class _LogC3:
    """ARRI LogC3 (EI 800) HDR compression constants and inverse.

    compress(): linear [0, ∞) → [-1, 1] (what the VAE learns on)
    decompress(): [-1, 1] → linear [0, ∞) (actual HDR values)
    """

    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    @classmethod
    def decompress(cls, z: torch.Tensor) -> torch.Tensor:
        # Undo the [0, 1] → [-1, 1] affine the model trained on.
        logc = torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)
        # Piecewise inverse: logarithmic above CUT, linear below.
        cut_log = cls.E * cls.CUT + cls.F
        lin_from_log = (torch.pow(10.0, (logc - cls.D) / cls.C) - cls.B) / cls.A
        lin_from_lin = (logc - cls.F) / cls.E
        return torch.where(logc >= cut_log, lin_from_log, lin_from_lin)


class RSLogC3Decode:
    """Decode LogC3 HDR frames from the LTX-2.3 HDR IC-LoRA into linear HDR
    (for EXR save) plus a tonemapped SDR preview (for ProRes/MP4 save)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "LogC3-encoded frames straight from VAE Decode after the HDR IC-LoRA. Values in [0, 1]."}),
            },
            "optional": {
                "exposure_stops": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                                              "tooltip": "Exposure adjustment applied to the SDR preview only (in stops, ±EV). The hdr_linear and raw outputs are left alone."}),
                "hdr_clamp_max": ("FLOAT", {"default": 10000.0, "min": 1.0, "max": 100000.0, "step": 10.0,
                                             "tooltip": "Upper bound applied to the decompressed HDR values. 10000 matches the reference implementation (HDR10 peak nits)."}),
                "preview_gamma": (["srgb", "linear"], {"default": "srgb",
                                                        "tooltip": "Color space for the SDR preview. srgb = standard display encoding (recommended for ProRes/MP4). linear = raw tonemapped values, use when feeding into a viewer that expects linear."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("hdr_linear", "raw", "sdr_preview")
    FUNCTION = "decode"
    CATEGORY = "rs-nodes"

    @staticmethod
    def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
        """Piecewise sRGB gamma (IEC 61966-2-1)."""
        x = torch.clamp(x, 0.0, 1.0)
        a = 0.055
        lin_part = x * 12.92
        gamma_part = (1 + a) * torch.pow(x, 1.0 / 2.4) - a
        return torch.where(x <= 0.0031308, lin_part, gamma_part)

    def decode(self, images, exposure_stops=0.0, hdr_clamp_max=10000.0, preview_gamma="srgb"):
        # raw = VAE output passthrough, the LogC3-encoded frames in [0, 1].
        # Use this when handing frames to an external grading pipeline that
        # wants to do its own LogC → linear conversion.
        raw = images[..., :3].clone()

        # hdr_linear = scene-referred linear HDR. Invert the LogC3 curve,
        # clamp to a sane upper bound. This is what EXR consumers expect.
        z = raw * 2.0 - 1.0  # LogC3's internal [-1, 1] domain
        hdr_linear = torch.clamp(_LogC3.decompress(z), 0.0, hdr_clamp_max)

        # sdr_preview = tonemapped + gamma'd for display. Always derived from
        # the linear HDR so it looks right regardless of which output you
        # actually save.
        exposed = hdr_linear * (2.0 ** exposure_stops)
        tonemapped = exposed / (1.0 + exposed)
        if preview_gamma == "srgb":
            preview = self._linear_to_srgb(tonemapped)
        else:
            preview = torch.clamp(tonemapped, 0.0, 1.0)

        logger.info(
            f"LogC3 decode: {hdr_linear.shape[0]} frame(s), "
            f"linear HDR range [{hdr_linear.min().item():.3f}, {hdr_linear.max().item():.3f}], "
            f"preview exposure={exposure_stops:+.2f} stops, gamma={preview_gamma}"
        )
        return (hdr_linear, raw, preview)
