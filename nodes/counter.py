import json
import logging
import os

import folder_paths

logger = logging.getLogger(__name__)

_STATE_FILENAME = "counter_state.json"


def _state_path() -> str:
    return os.path.join(folder_paths.get_output_directory(), _STATE_FILENAME)


def _load_state() -> dict:
    path = _state_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"RSCounter: could not read state file, starting fresh ({e})")
    return {"value": 0}


def _save_state(state: dict) -> None:
    path = _state_path()
    with open(path, "w") as f:
        json.dump(state, f)


class RSCounter:
    """Persistent counter node. Outputs the current value then increments by step.
    State is stored in counter_state.json inside ComfyUI's output directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0}),
                "step":  ("INT", {"default": 1}),
                "reset": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, start: int, step: int, reset: bool):
        state = _load_state()

        if reset:
            state["value"] = start
            logger.info(f"RSCounter: reset to {start}")

        current = state["value"]
        state["value"] = current + step
        _save_state(state)

        logger.info(f"RSCounter: output {current}, next will be {state['value']}")
        return (current,)
