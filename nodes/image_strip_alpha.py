"""Strip the alpha channel from an IMAGE tensor.

ComfyUI's vanilla CreateVideo / SaveVideo (comfy_extras/nodes_video.py)
encodes frames via `av.VideoFrame.from_ndarray(img, format='rgb24')`,
which hard-rejects any array whose last axis isn't 3. Several nodes —
SAM3 mask compositing, Depth Anything's coloured depth output, etc. —
emit 4-channel RGBA images, and the SaveVideo step blows up with
"Unexpected numpy array shape (H, W, 4)".

This node sits between the offending source and the video saver and
flattens the alpha channel by compositing onto a chosen background
colour (default black). 3-channel inputs pass through untouched, so
it's safe to leave wired in even when the upstream source is already
RGB.
"""

import torch


class RSImageStripAlpha:
    """Drop the alpha channel from an IMAGE, compositing onto a
    background colour. Pass-through when the input is already RGB."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "background": (
                    ["black", "white", "transparent_to_black"],
                    {
                        "default": "black",
                        "tooltip": (
                            "Colour the alpha is composited onto. "
                            "'transparent_to_black' premultiplies — same "
                            "as 'black' but explicit about the math."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "strip"
    CATEGORY = "rs-nodes"

    def strip(self, image, background="black"):
        # Comfy IMAGE tensors are [B, H, W, C] float32 in 0-1. We only
        # need to act when C == 4; 3-channel inputs pass straight
        # through so this node is safe to leave wired in always.
        if image.ndim != 4 or image.shape[-1] == 3:
            return (image,)
        if image.shape[-1] != 4:
            # Unexpected channel count (1 / 2 / >4). Strip down to the
            # first three planes and let the caller deal — better than
            # failing here.
            return (image[..., :3].contiguous(),)

        rgb = image[..., :3]
        alpha = image[..., 3:4]  # keep the trailing axis so broadcasting works

        if background == "white":
            bg_value = 1.0
        else:
            bg_value = 0.0

        # Standard "over" compositing: out = rgb * alpha + bg * (1 - alpha)
        composited = rgb * alpha + bg_value * (1.0 - alpha)
        # Clamp + force contiguous so downstream av.from_ndarray is
        # happy (it requires a C-contiguous memory layout).
        composited = composited.clamp(0.0, 1.0).contiguous()
        return (composited,)
