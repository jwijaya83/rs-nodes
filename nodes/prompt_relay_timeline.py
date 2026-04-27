"""RSPromptRelayTimeline — canvas-driven JSON builder for Prompt Relay.

Produces the JSON schema consumed by RSPromptRelayEncode. The actual UI lives
in web/prompt_relay_timeline.js (custom canvas widget); the Python side is a
pass-through that emits whatever JSON the JS widget has authored into the
hidden `timeline_data` STRING.

Optionally accepts an audio file (uploaded via the standard ComfyUI file
picker) and emits an AUDIO output for downstream nodes (e.g. RSLTXVGenerate).
The same file is fetched by the JS side via /api/view to render the waveform
on the timeline track and provide playback for syncing segment edits to audio.

See Reference/prompt-relay-timeline-plan.md for the build plan.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import folder_paths

logger = logging.getLogger(__name__)


def _load_audio_file(filepath: str) -> tuple[torch.Tensor, int]:
    """Decode an audio file to (waveform[C, S], sample_rate) using PyAV.

    Mirrors comfy_extras/nodes_audio.py:load() so we get the same semantics
    as the standard LoadAudio node.
    """
    import av  # PyAV — ships with ComfyUI's deps

    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        if not wav.dtype.is_floating_point:
            if wav.dtype == torch.int16:
                wav = wav.float() / (2 ** 15)
            elif wav.dtype == torch.int32:
                wav = wav.float() / (2 ** 31)
            else:
                wav = wav.float()
        return wav, sr


def _silent_audio() -> dict:
    """Fallback AUDIO value for when no file is selected."""
    return {"waveform": torch.zeros(1, 1, 1, dtype=torch.float32), "sample_rate": 16000}


class RSPromptRelayTimeline:
    """Canvas timeline builder. Outputs a JSON STRING for RSPromptRelayEncode,
    plus AUDIO passthrough when an audio clip is loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            input_dir = folder_paths.get_input_directory()
            audio_files = folder_paths.filter_files_content_types(
                os.listdir(input_dir), ["audio", "video"]
            )
            audio_files = sorted(audio_files)
        except Exception:
            audio_files = []
        # Always include a sentinel "(none)" so users can clear the selection.
        audio_choices = ["(none)"] + audio_files
        return {
            "required": {
                "total_duration_sec": ("FLOAT", {"default": 4.0,  "min": 0.1, "max": 600.0,  "step": 0.01,
                                                  "tooltip": "Total clip length the timeline spans (seconds)."}),
                "frame_rate":         ("FLOAT", {"default": 25.0, "min": 0.1, "max": 1000.0, "step": 0.01,
                                                  "tooltip": "Frame rate; segment edges snap to 1/fps."}),
                "audio_file":         (audio_choices, {"audio_upload": True,
                                                       "tooltip": "Audio clip to align segments against. Plays back on the timeline."}),
                "audio_in_sec":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.001,
                                                  "tooltip": "Audio in-point (seconds). Set by the timeline trim handle."}),
                "audio_out_sec":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.001,
                                                  "tooltip": "Audio out-point (seconds). 0 = use full clip."}),
                "timeline_data":      ("STRING", {"default": "", "multiline": True,
                                                   "tooltip": "JSON authored by the canvas widget. Editable as a fallback."}),
            },
            "optional": {
                "style": ("STRING", {"default": "", "forceInput": True,
                                      "tooltip": "Optional override for the style/global prompt. When wired, replaces whatever is typed in the canvas Style box."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "AUDIO")
    RETURN_NAMES = ("relay_json", "num_frames", "frame_rate", "audio")
    FUNCTION = "build"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = False

    @staticmethod
    def _legal_num_frames(total_duration_sec: float, frame_rate: float) -> int:
        """Round total_duration_sec * frame_rate to the nearest LTX-legal frame
        count, i.e. (N - 1) % 8 == 0 and N >= 9."""
        raw = max(int(round(total_duration_sec * frame_rate)), 1)
        k = round((raw - 1) / 8.0)
        return max(8 * k + 1, 9)

    def build(self, total_duration_sec, frame_rate, audio_file,
              audio_in_sec, audio_out_sec, timeline_data, style=""):
        s = (timeline_data or "").strip()
        if not s:
            s = "{}"
        # If a `style` override is wired in, splice it into the JSON's global field.
        style = (style or "").strip()
        if style:
            try:
                data = json.loads(s)
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}
            data["global"] = style
            data.setdefault("segments", [])
            s = json.dumps(data, indent=2)
        num_frames = self._legal_num_frames(float(total_duration_sec), float(frame_rate))

        audio = _silent_audio()
        if audio_file and audio_file != "(none)":
            try:
                path = folder_paths.get_annotated_filepath(audio_file)
                wav, sr = _load_audio_file(path)  # wav: [C, S]
                in_sec = max(0.0, float(audio_in_sec))
                out_sec = float(audio_out_sec)
                if out_sec <= in_sec:
                    out_sec = wav.shape[-1] / float(sr)  # full clip
                in_sample = max(0, int(round(in_sec * sr)))
                out_sample = min(int(round(out_sec * sr)), wav.shape[-1])
                if out_sample > in_sample:
                    wav = wav[:, in_sample:out_sample].contiguous()
                audio = {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}
            except Exception as e:
                logger.warning(f"RSPromptRelayTimeline: failed to load audio '{audio_file}': {e}")

        return (s, num_frames, float(frame_rate), audio)
