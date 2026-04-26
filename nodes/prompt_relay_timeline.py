"""RSPromptRelayTimeline — canvas-driven JSON builder for Prompt Relay.

Produces the JSON schema consumed by RSPromptRelayEncode. The actual UI lives
in web/prompt_relay_timeline.js (custom canvas widget); the Python side is a
pass-through that emits whatever JSON the JS widget has authored into the
hidden `timeline_data` STRING.

Layered architecture: this node only writes JSON, the encoder only consumes
JSON. Hand-typed JSON in the encoder is still valid; an Ollama formatter node
could likewise emit JSON and feed the encoder.

See Reference/prompt-relay-timeline-plan.md for the build plan.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RSPromptRelayTimeline:
    """Canvas timeline builder. Outputs a JSON STRING for RSPromptRelayEncode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_duration_sec": ("FLOAT", {"default": 4.0,  "min": 0.1, "max": 600.0,  "step": 0.01,
                                                  "tooltip": "Total clip length the timeline spans (seconds)."}),
                "frame_rate":         ("FLOAT", {"default": 25.0, "min": 0.1, "max": 1000.0, "step": 0.01,
                                                  "tooltip": "Frame rate; segment edges snap to 1/fps."}),
                "timeline_data":      ("STRING", {"default": "", "multiline": True,
                                                   "tooltip": "JSON authored by the canvas widget. Editable as a fallback."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relay_json",)
    FUNCTION = "build"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = False

    def build(self, total_duration_sec, frame_rate, timeline_data):
        s = (timeline_data or "").strip()
        if not s:
            s = "{}"
        return (s,)
