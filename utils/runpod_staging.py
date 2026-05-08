"""RunPod-bound save staging.

When ComfyUI runs on a RunPod pod and the user wants the high-quality
output (ProRes / EXR sequence) ultimately to land on their LOCAL
filesystem, the save node can't push to local directly — it can only
write to the pod's filesystem. This module standardises a staging
convention so a separate local-side pull tool can reconstruct what
goes where.

Layout on the pod (under ComfyUI's output dir):

    output/
      runpod_pending/
        <run_id>/                # one save per run_id
          .manifest.json         # tells the puller where the user
                                 # wants the contents to land locally
          <whatever the saver writes — single .mov, an EXR sequence
           directory, etc.>

The manifest schema is intentionally tiny so the pull script stays
dumb. Adding fields is fine; readers ignore unknowns.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Literal

logger = logging.getLogger(__name__)

MANIFEST_NAME = ".manifest.json"
PENDING_SUBDIR = "runpod_pending"


@dataclass
class StagingHandle:
    """One save's staging area. The save node writes its files into
    `staging_dir`, then calls write_manifest() to mark it ready."""
    run_id: str
    staging_dir: str

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.staging_dir, MANIFEST_NAME)


def staging_root() -> str:
    """The directory all pending pulls accumulate under.

    Lazy-imports folder_paths so this module is importable outside a
    running ComfyUI (for unit tests etc.).
    """
    import folder_paths  # type: ignore
    base = os.path.join(folder_paths.get_output_directory(), PENDING_SUBDIR)
    os.makedirs(base, exist_ok=True)
    return base


def allocate(prefix_hint: str = "") -> StagingHandle:
    """Reserve a staging directory for one save operation.

    `prefix_hint` is appended to the run_id only as a debugging aid so
    a human eyeballing the pod's pending dir can guess what's what.
    Pull tools key off the manifest, never the directory name.
    """
    run_id = uuid.uuid4().hex[:12]
    if prefix_hint:
        # Sanitize — directory names get keyboard-friendly chars only.
        clean = "".join(c if c.isalnum() or c in "._-" else "_" for c in prefix_hint)
        run_id = f"{clean[:32]}_{run_id}"
    staging_dir = os.path.join(staging_root(), run_id)
    os.makedirs(staging_dir, exist_ok=True)
    return StagingHandle(run_id=run_id, staging_dir=staging_dir)


DEFAULT_PULL_SERVER_URL = "http://localhost:8765/pull"


def write_manifest(
    handle: StagingHandle,
    *,
    target_path: str,
    kind: Literal["file", "directory"] = "file",
    saver: str = "",
    extra: dict | None = None,
) -> None:
    """Mark the staging dir as ready for the puller to pick up.

    target_path is interpreted on the LOCAL machine. It can be either
    a directory (puller appends the staged filename) or a full file
    path (puller renames if the names differ). The saver writes the
    file under handle.staging_dir; the puller does the placement.

    Writing the manifest LAST is intentional — the puller skips
    staging dirs that don't have a manifest yet, so partial saves
    can't be picked up.
    """
    manifest = {
        "target_path": target_path,
        "kind": kind,
        "saver": saver,
        "saved_at": time.time(),
    }
    if extra:
        manifest.update(extra)
    # Write to a temp name then rename so a half-written manifest
    # never appears to the puller.
    tmp = handle.manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    os.replace(tmp, handle.manifest_path)
    logger.info(
        f"runpod_staging: marked {handle.run_id} ready -> {target_path}"
    )


def notify_pull(
    handle: StagingHandle,
    *,
    target_path: str,
    kind: Literal["file", "directory"] = "file",
    callback_url: str = DEFAULT_PULL_SERVER_URL,
    timeout: float = 10.0,
) -> bool:
    """POST to a local pull-server with the info to pull this save.

    The save node, running on the pod, hits `callback_url`; on a
    correctly-set-up session that's a localhost address forwarded via
    SSH reverse tunnel back to the user's Windows machine, where
    pull_server.py listens and runs an outbound `scp` from the pod.

    Failures are logged but do NOT raise — a flaky callback shouldn't
    fail the save itself. The manifest is still on the pod, so a
    later manual pull always recovers the file.

    Returns True on a 2xx response, False otherwise.
    """
    if not callback_url or not callback_url.strip():
        return False

    # Find the actual saved file/dir inside the staging dir
    # (the puller needs the full path on the pod).
    try:
        contents = [
            os.path.join(handle.staging_dir, n)
            for n in os.listdir(handle.staging_dir)
            if n != MANIFEST_NAME and not n.endswith(".tmp")
        ]
    except OSError as e:
        logger.warning(f"notify_pull: cannot list {handle.staging_dir}: {e}")
        return False

    if not contents:
        logger.warning(
            f"notify_pull: nothing to pull in {handle.staging_dir} "
            "(saver wrote no file?)"
        )
        return False

    # First entry — there should only be one for a single save.
    remote_path = contents[0]

    payload = {
        "remote_path": remote_path,
        "local_path": target_path,
        "kind": kind,
        "run_id": handle.run_id,
        "staging_dir": handle.staging_dir,
    }

    # Use stdlib only — no extra dep on requests for the pod side.
    from urllib import error as urlerror, request as urlrequest

    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        callback_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            if ok:
                logger.info(
                    f"runpod_staging: pull-server notified for {handle.run_id}"
                )
            else:
                logger.warning(
                    f"runpod_staging: pull-server returned {resp.status} "
                    f"for {handle.run_id}"
                )
            return ok
    except (urlerror.URLError, OSError) as e:
        logger.warning(
            f"runpod_staging: pull-server callback failed: {e}. "
            f"Manifest still in {handle.staging_dir}; pull manually."
        )
        return False
