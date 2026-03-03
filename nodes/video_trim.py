import torch


class RSVideoTrim:
    """Trim a video frame batch by time (seconds)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "in_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 0.01}),
                "out_point": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 9999.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("images", "fps", "frame_count")
    FUNCTION = "trim"
    CATEGORY = "rs-nodes"

    def trim(self, images: torch.Tensor, fps: float, in_point: float, out_point: float):
        total_frames = images.shape[0]

        start_frame = round(in_point * fps)
        # out_point == 0.0 means "use end of video"
        end_frame = total_frames if out_point == 0.0 else round(out_point * fps)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        if start_frame >= end_frame:
            # Return a single black frame at the input resolution
            H, W, C = images.shape[1], images.shape[2], images.shape[3]
            result = torch.zeros((1, H, W, C), dtype=images.dtype, device=images.device)
            print(f"[RSVideoTrim] Warning: in_point >= out_point after clamping — returning single black frame")
            return (result, fps, 1)

        result = images[start_frame:end_frame]
        count = result.shape[0]

        print(
            f"[RSVideoTrim] Trimmed: {in_point:.2f}s-{out_point:.2f}s "
            f"(frames {start_frame}-{end_frame}, {count} frames)"
        )

        return (result, fps, count)
