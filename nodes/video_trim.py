import torch


class RSVideoTrim:
    """Trim video frames and/or audio by time (seconds)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "in_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 0.01}),
                "out_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count")
    FUNCTION = "trim"
    CATEGORY = "rs-nodes"

    def trim(self, fps, in_point, out_point, images=None, audio=None):
        trimmed_images = None
        trimmed_audio = None
        frame_count = 0

        # --- Trim images ---
        if images is not None:
            total_frames = images.shape[0]
            start_frame = round(in_point * fps)
            end_frame = total_frames if out_point == 0.0 else round(out_point * fps)

            start_frame = max(0, min(start_frame, total_frames))
            end_frame = max(0, min(end_frame, total_frames))

            if start_frame >= end_frame:
                H, W, C = images.shape[1], images.shape[2], images.shape[3]
                trimmed_images = torch.zeros((1, H, W, C), dtype=images.dtype, device=images.device)
                frame_count = 1
                print(f"[RSVideoTrim] Warning: in_point >= out_point after clamping — returning single black frame")
            else:
                trimmed_images = images[start_frame:end_frame]
                frame_count = trimmed_images.shape[0]
                print(f"[RSVideoTrim] Images trimmed: {in_point:.2f}s-{out_point:.2f}s "
                      f"(frames {start_frame}-{end_frame}, {frame_count} frames)")

        # --- Trim audio ---
        if audio is not None:
            waveform = audio["waveform"]  # (B, C, S)
            sample_rate = audio["sample_rate"]

            total_samples = waveform.shape[-1]
            start_sample = round(in_point * sample_rate)
            end_sample = total_samples if out_point == 0.0 else round(out_point * sample_rate)

            start_sample = max(0, min(start_sample, total_samples))
            end_sample = max(0, min(end_sample, total_samples))

            if start_sample >= end_sample:
                trimmed_waveform = torch.zeros((waveform.shape[0], waveform.shape[1], 1),
                                               dtype=waveform.dtype, device=waveform.device)
                print(f"[RSVideoTrim] Warning: audio in_point >= out_point — returning silence")
            else:
                trimmed_waveform = waveform[..., start_sample:end_sample]
                duration = trimmed_waveform.shape[-1] / sample_rate
                print(f"[RSVideoTrim] Audio trimmed: {in_point:.2f}s-{out_point:.2f}s "
                      f"(samples {start_sample}-{end_sample}, {duration:.2f}s)")

            trimmed_audio = {"waveform": trimmed_waveform, "sample_rate": sample_rate}

            # If no images provided, derive frame_count from audio duration
            if images is None:
                duration = trimmed_waveform.shape[-1] / sample_rate
                frame_count = round(duration * fps)

        return (trimmed_images, trimmed_audio, fps, frame_count)
