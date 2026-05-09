"""Workflow asset detection + JSON rewrite.

When the user dispatches a graph to a remote pod, every local file the
graph references has to make it to the pod's `input/` directory before
the workflow can execute, and the workflow JSON has to be rewritten so
those references point at the remote names. This module does both.

Two-pass strategy:

1. Known input-node registry. For each node class we explicitly know
   about (LoadImage, LoadAudio, etc., plus rs-nodes additions), we list
   exactly which input keys are file references. This is precise and
   doesn't risk false positives.

2. Heuristic fallback. For node classes not in the registry, any STRING
   input whose value resolves to an existing local file (and isn't on
   the deny-list) is treated as an asset. The heuristic does NOT fire
   for known node classes — those are fully covered by the registry,
   so heuristic + registry can't both flag the same input.

Returned manifest entries carry enough info for the dispatcher to:
  - upload the file (local_path, remote_name, size)
  - report progress meaningfully (description = "node_id.input_name")
  - skip re-uploads on rapid re-dispatch (sha256_short in remote_name)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .runpod_client import file_sha256_short

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry of known input nodes
# ---------------------------------------------------------------------------
# Each entry: class_type -> { input_name: kind }
#
# kind:
#   "comfy_input"  - filename relative to ComfyUI's input/ dir. Resolve
#                    via folder_paths, upload, keep the same remote name.
#   "abs_file"     - absolute path to a local file. Upload, rewrite the
#                    JSON to use the remote-stored name (unqualified).
#   "abs_dir"      - directory path. We REFUSE to auto-upload (too big
#                    for /upload/image; deferred to bulk-upload tool).
#                    Surfaces a clear error pointing the user there.
#
# Anything not listed here falls through to the heuristic pass.

KNOWN_FILE_INPUTS: dict[str, dict[str, str]] = {
    # Stock ComfyUI nodes
    "LoadImage": {"image": "comfy_input"},
    "LoadImageMask": {"image": "comfy_input"},
    "LoadImageOutput": {"image": "comfy_input"},
    "LoadAudio": {"audio": "comfy_input"},
    "LoadVideo": {"video": "comfy_input"},
    "VHS_LoadVideo": {"video": "comfy_input"},
    "VHS_LoadAudio": {"audio": "comfy_input"},

    # rs-nodes additions — directory inputs (training data); refused
    # by v1, surfaced via clear error pointing at the bulk-upload tool.
    "RSLTXVPrepareDataset": {"media_folder": "abs_dir"},
    "RSLTXVTrainLoRA": {"dataset_dir": "abs_dir"},
}


# Extensions we'll consider for the heuristic pass. Restricting to media
# files keeps us from accidentally trying to upload, say, a model name
# string that happens to exist as a file somewhere weird.
HEURISTIC_FILE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".exr",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
    ".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a",
    ".safetensors", ".ckpt", ".pt", ".bin",  # included for completeness;
    # callers should keep models on the network volume rather than
    # uploading via /upload/image. The size cap will block the big ones.
    ".json", ".txt", ".csv",  # captions / lists
    ".npy", ".npz",
}


# Refuse to even consider these path prefixes as upload candidates.
# Catches "C:\Windows\System32\..." style false positives from STRING
# inputs that happen to spell a real system file.
DENY_PATH_PREFIXES = (
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/proc/",
    "/sys/",
)


# ---------------------------------------------------------------------------
# Manifest types
# ---------------------------------------------------------------------------

@dataclass
class AssetEntry:
    """A single local file that needs to make the trip to the pod."""
    node_id: str
    class_type: str
    input_name: str
    local_path: Path
    size_bytes: int
    sha256_short: str
    remote_name: str
    # The original value as it appeared in the workflow JSON (filename
    # or absolute path). Useful when reverse-mapping after the run.
    original_value: str

    @property
    def description(self) -> str:
        return f"{self.class_type}#{self.node_id}.{self.input_name}"


@dataclass
class AssetPlan:
    """Result of a workflow walk."""
    rewritten_workflow: dict
    assets: list[AssetEntry] = field(default_factory=list)
    # Detected directory inputs — populated only on error so the
    # dispatcher can surface them with the bulk-upload pointer.
    rejected_directories: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(a.size_bytes for a in self.assets)


class AssetWalkError(RuntimeError):
    """Raised when the workflow can't be prepared for remote dispatch."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_rewrite(
    workflow: dict,
    *,
    comfy_input_dir: Path | None = None,
    max_upload_bytes: int | None = None,
) -> AssetPlan:
    """Walk a workflow (API-format prompt dict), produce an AssetPlan.

    Args:
        workflow: ComfyUI prompt-format dict — `{node_id: {class_type,
            inputs}}`. NOT the GUI workflow dict.
        comfy_input_dir: Override for ComfyUI's input dir. If None,
            resolves via folder_paths (lazy import). Used to look up
            files referenced by `LoadImage`-style "comfy_input" inputs.
        max_upload_bytes: If set, raise AssetWalkError when any single
            file exceeds this. The dispatcher hands in the same value
            it'll enforce on upload, so the user gets the size error
            up-front rather than mid-upload.

    Returns:
        AssetPlan with rewritten_workflow ready to submit and an asset
        list ready to upload.

    Raises:
        AssetWalkError: a directory input was hit, or a file exceeded
            max_upload_bytes. Message names the offending node.
    """
    if not isinstance(workflow, dict):
        raise AssetWalkError("workflow must be a dict")

    if comfy_input_dir is None:
        comfy_input_dir = _resolve_comfy_input_dir()

    rewritten: dict = {}
    assets: list[AssetEntry] = []
    rejected_dirs: list[tuple[str, str, str]] = []

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            rewritten[node_id] = node
            continue

        class_type = node.get("class_type", "")
        inputs = node.get("inputs") or {}
        new_inputs = dict(inputs)

        known = KNOWN_FILE_INPUTS.get(class_type, {})

        for input_name, value in inputs.items():
            kind = known.get(input_name)
            if kind is None:
                # Heuristic only fires for unknown node classes so
                # registered nodes never get double-processed.
                if class_type in KNOWN_FILE_INPUTS:
                    continue
                kind = _heuristic_kind(value)
                if kind is None:
                    continue

            local_path = _resolve_input_to_path(
                kind, value, comfy_input_dir,
            )
            if local_path is None:
                # Registry said it's a file ref but it doesn't exist
                # locally — pass the value through unchanged. The
                # remote may have it staged on the network volume.
                continue

            if kind == "abs_dir" or local_path.is_dir():
                rejected_dirs.append((node_id, class_type, str(local_path)))
                continue

            try:
                size = local_path.stat().st_size
            except OSError as e:
                logger.warning(
                    f"Could not stat {local_path} for "
                    f"{class_type}#{node_id}.{input_name}: {e}"
                )
                continue

            if max_upload_bytes is not None and size > max_upload_bytes:
                raise AssetWalkError(
                    f"{class_type}#{node_id}.{input_name} -> "
                    f"{local_path.name} is {size / 1024**3:.1f} GB, "
                    f"exceeds the {max_upload_bytes / 1024**3:.0f} GB "
                    "inline-upload limit. Stage this file on the "
                    "network volume directly, or wait for the bulk-"
                    "upload tool (deferred follow-up)."
                )

            sha = file_sha256_short(local_path)
            remote_name = f"{sha}_{local_path.name}"

            assets.append(AssetEntry(
                node_id=node_id,
                class_type=class_type,
                input_name=input_name,
                local_path=local_path,
                size_bytes=size,
                sha256_short=sha,
                remote_name=remote_name,
                original_value=str(value),
            ))

            new_inputs[input_name] = remote_name

        # Replace the node's inputs in the rewritten copy. We keep
        # everything else (class_type, _meta, etc.) intact.
        rewritten_node = dict(node)
        rewritten_node["inputs"] = new_inputs
        rewritten[node_id] = rewritten_node

    if rejected_dirs:
        details = "\n".join(
            f"  - {ct}#{nid}: {p}"
            for nid, ct, p in rejected_dirs
        )
        raise AssetWalkError(
            "Workflow references local directories. Directory inputs "
            "(training datasets etc.) are not auto-uploaded — stage "
            "them on the network volume directly.\n"
            f"Offending nodes:\n{details}"
        )

    return AssetPlan(
        rewritten_workflow=rewritten,
        assets=assets,
        rejected_directories=rejected_dirs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_comfy_input_dir() -> Path:
    """Resolve ComfyUI's input/ directory via folder_paths.

    Lazy-imported because workflow_assets.py is imported at node-load
    time and we don't want a hard dep on folder_paths just to be
    importable. By the time detect_and_rewrite() actually runs the
    dispatcher is inside ComfyUI's runtime so the import succeeds.
    """
    try:
        import folder_paths  # type: ignore
    except ImportError:
        # Fallback: best-guess relative to this file's location.
        return Path(__file__).resolve().parents[3] / "input"
    return Path(folder_paths.get_directory_by_type("input"))


def _heuristic_kind(value: Any) -> str | None:
    """Decide whether a STRING value looks like a local file ref.

    Returns "abs_file" if so, "abs_dir" if it's a directory, None
    otherwise. Cheap checks first to avoid hammering the FS for every
    string in every workflow.
    """
    if not isinstance(value, str):
        return None
    if not value or len(value) > 1024:  # absurdly long strings aren't paths
        return None

    # Cheap rejects before touching the filesystem.
    for prefix in DENY_PATH_PREFIXES:
        if value.startswith(prefix):
            return None

    # Path-shaped check: contains a separator AND has an extension
    # OR is clearly absolute. Avoids false positives on prompt strings.
    looks_like_path = (
        os.sep in value
        or "/" in value
        or (len(value) > 2 and value[1] == ":")  # Windows drive letter
    )
    if not looks_like_path:
        return None

    ext = os.path.splitext(value)[1].lower()
    if ext and ext not in HEURISTIC_FILE_EXTS:
        # Has an extension but not one we recognize — skip. (This is
        # what stops, e.g., a ".pt" path on the network volume from
        # being mistaken for a local upload candidate when the file
        # also happens to exist locally.)
        if ext not in HEURISTIC_FILE_EXTS:
            pass
        # Falls through to existence check anyway for extension-less
        # files; for extensioned files we bail.
        return None

    p = Path(value)
    try:
        if p.is_file():
            return "abs_file"
        if p.is_dir():
            return "abs_dir"
    except OSError:
        return None
    return None


def _resolve_input_to_path(
    kind: str,
    value: Any,
    comfy_input_dir: Path,
) -> Path | None:
    """Resolve a (kind, value) into a concrete local Path, or None
    if the file isn't where the registry said it should be."""
    if not isinstance(value, str) or not value:
        return None

    if kind == "comfy_input":
        # A filename relative to ComfyUI's input dir. ComfyUI also
        # supports a "subfolder" via "subfolder/file.png" in the
        # widget; preserve that.
        candidate = comfy_input_dir / value
        if candidate.is_file():
            return candidate
        # ComfyUI's LoadImage widget sometimes stores
        # "[input] file.png" or similar — not handled here. If we
        # encounter it the file just passes through unchanged.
        return None

    if kind in ("abs_file", "abs_dir"):
        p = Path(value)
        if p.exists():
            return p
        return None

    return None


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def output_dir_for_run(local_output_root: Path, prompt_id: str) -> Path:
    """Where to drop a run's downloaded outputs.

    `local_output_root` is ComfyUI's local output/ dir; we put each
    run in its own subdir so concurrent dispatches don't collide.
    """
    target = local_output_root / "runpod" / prompt_id
    target.mkdir(parents=True, exist_ok=True)
    return target
