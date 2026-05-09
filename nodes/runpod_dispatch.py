"""RSRunOnRunPod — dispatch the surrounding workflow to a RunPod pod.

The user puts this single node into a graph that contains heavy
nodes which would OOM on the local 16 GB GPU. They click the
"Run on RunPod" button (added by web/runpod_dispatch.js) on this
node. The button captures the rest of the graph as JSON, ferries
it into the dispatcher's hidden _workflow_json input, and queues
a single-node prompt that runs ONLY this node locally. This
dispatcher then:

  1. Resolves credentials for the chosen profile.
  2. Starts (or reuses) a pod on the user's network volume.
  3. Polls until the remote ComfyUI is up.
  4. Walks the captured workflow JSON and uploads any local file
     references via /upload/image, rewriting the JSON to use the
     remote-stored names.
  5. POSTs the rewritten workflow to /prompt.
  6. Streams progress events back to the local UI (per-step
     progress bar + phase log on the dispatcher node body).
  7. Downloads the produced outputs into output/runpod/<id>/.
  8. Stops the pod (unless auto_stop=False).

If the user presses ComfyUI's interrupt during a remote run, the
dispatcher posts /interrupt to the pod, then proceeds to stop_pod
so the user isn't billed for a zombie.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from ..utils import runpod_client
from ..utils import workflow_assets
from ..utils.runpod_credentials import RunpodCredsError, resolve as resolve_creds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU options surfaced in the dropdown
# ---------------------------------------------------------------------------
# These are RunPod GPU type IDs. The list is intentionally short — the
# user can edit it as their account's available GPUs change. RunPod's
# /gpuTypes API would be more dynamic but adds a startup roundtrip; a
# static list keeps the node load fast.
GPU_TYPES = [
    "NVIDIA RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
]


# ---------------------------------------------------------------------------
# Cancel-aware helper
# ---------------------------------------------------------------------------

def _make_cancel_check():
    """Return a callable that raises RuntimeError if ComfyUI's
    interrupt button has been pressed. Lazy-imports model_management
    so this module is importable outside ComfyUI for unit tests."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return lambda: None

    def check():
        try:
            mm.throw_exception_if_processing_interrupted()
        except Exception:
            raise

    return check


# ---------------------------------------------------------------------------
# Frontend event relay
# ---------------------------------------------------------------------------

EVENT_PHASE = "rs.runpod.phase"
EVENT_PROGRESS = "rs.runpod.progress"
EVENT_LOG = "rs.runpod.log"


def _send(event: str, node_id: str | None, payload: dict) -> None:
    """Push a status event to the JS extension. Silent on failure
    — progress notifications are best-effort, never load-bearing."""
    try:
        from server import PromptServer  # type: ignore
    except ImportError:
        return
    if PromptServer is None or PromptServer.instance is None:
        return
    body = dict(payload)
    if node_id is not None:
        body["node_id"] = str(node_id)
    try:
        PromptServer.instance.send_sync(event, body)
    except Exception as e:
        logger.debug(f"send_sync({event}) failed: {e}")


def _phase(node_id: str | None, step: int, total: int, text: str) -> None:
    """Phase-log message: '[step/total] text'. Mirrors to local logger."""
    line = f"[{step}/{total}] {text}"
    logger.info(line)
    _send(EVENT_PHASE, node_id, {"step": step, "total": total, "text": text, "line": line})


def _log(node_id: str | None, text: str) -> None:
    logger.info(text)
    _send(EVENT_LOG, node_id, {"text": text})


# ---------------------------------------------------------------------------
# Comfy progress-bar bridge
# ---------------------------------------------------------------------------

class _RemoteProgressBridge:
    """Translate remote WebSocket events into local UI updates.

    Two layers:
      - Per-step progress bar: forwarded to the dispatcher node's
        progress widget via PromptServer's "progress" event so the
        top-of-screen progress bar fills as the remote sampler ticks.
      - Per-node phase log: each "executing" event prints the new
        node class to the dispatcher's status panel.
    """

    def __init__(self, node_id: str | None, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = max(1, total_nodes)
        self.completed_nodes = 0
        self.current_node_id: str | None = None
        # Remember the most recent (max) we saw so we can recompute
        # the bar fill ratio without per-event division on the JS side.
        self.last_value = 0
        self.last_max = 1

    def __call__(self, event: dict) -> None:
        etype = event.get("type")
        edata = event.get("data") or {}

        if etype == "executing":
            remote_node = edata.get("node")
            if remote_node is None:
                return
            if remote_node != self.current_node_id and self.current_node_id is not None:
                self.completed_nodes += 1
            self.current_node_id = remote_node
            _log(
                self.node_id,
                f"Remote: {remote_node} "
                f"({self.completed_nodes + 1}/{self.total_nodes})",
            )
            return

        if etype == "progress":
            value = int(edata.get("value", 0))
            mx = max(1, int(edata.get("max", 1)))
            self.last_value = value
            self.last_max = mx
            _send(EVENT_PROGRESS, self.node_id, {
                "value": value,
                "max": mx,
                "node": edata.get("node") or self.current_node_id,
                "completed_nodes": self.completed_nodes,
                "total_nodes": self.total_nodes,
            })
            return

        if etype == "executed":
            # A node finished. Don't increment here — "executing"
            # transitions handle that — but log saved outputs.
            output = edata.get("output") or {}
            if output:
                kinds = sorted(k for k in output.keys() if isinstance(output[k], list))
                if kinds:
                    _log(self.node_id, f"Output ready: {edata.get('node')} -> {','.join(kinds)}")


# ---------------------------------------------------------------------------
# Local output dir resolution
# ---------------------------------------------------------------------------

def _comfy_output_dir() -> Path:
    """Resolve ComfyUI's local output dir."""
    try:
        import folder_paths  # type: ignore
        return Path(folder_paths.get_output_directory())
    except ImportError:
        return Path(__file__).resolve().parents[3] / "output"


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class RSRunOnRunPod:
    """Dispatch the surrounding workflow to a RunPod pod.

    See module docstring for the high-level flow. The "Run on RunPod"
    button (added by web/runpod_dispatch.js) captures the graph and
    queues a one-node prompt containing only this dispatcher.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": ("STRING", {
                    "default": "default",
                    "tooltip": (
                        "Credentials profile name from "
                        "%USERPROFILE%/.runpod/credentials.ini. "
                        "Falls back to RUNPOD_PROFILE env var, then "
                        "RUNPOD_API_KEY+RUNPOD_USER_ID env vars."
                    ),
                }),
                "pod_template_id": ("STRING", {
                    "default": "",
                    "tooltip": "RunPod template ID for the pod image.",
                }),
                "network_volume_id": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Network-volume ID (mounts at /workspace). "
                        "Holds the persistent ComfyUI install + models."
                    ),
                }),
                "gpu_type": (GPU_TYPES, {
                    "default": "NVIDIA RTX 4090",
                }),
                "auto_stop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Stop the pod after the run completes. Disable "
                        "if you plan to dispatch again immediately."
                    ),
                }),
                "max_wait_minutes": ("INT", {
                    "default": 60, "min": 1, "max": 720,
                    "tooltip": "Maximum total time to wait for the run.",
                }),
                "capture_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "If true, request verbose remote logging "
                        "(future feature; currently a no-op flag "
                        "carried in the run summary)."
                    ),
                }),
            },
            "optional": {
                # Filled by the JS button before queueing. Not meant
                # to be edited by hand — the JS hides the widget.
                "_workflow_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("run_summary",)
    FUNCTION = "dispatch"
    OUTPUT_NODE = True
    CATEGORY = "rs-nodes/runpod"

    def dispatch(
        self,
        profile: str,
        pod_template_id: str,
        network_volume_id: str,
        gpu_type: str,
        auto_stop: bool,
        max_wait_minutes: int,
        capture_console: bool,
        _workflow_json: str = "",
        unique_id: str | None = None,
    ):
        node_id = unique_id
        run_started_at = time.time()
        cancel_check = _make_cancel_check()

        # ---- 1. Validate inputs ----
        if not _workflow_json.strip():
            raise RuntimeError(
                "No workflow captured. Click the 'Run on RunPod' button "
                "on this node — don't queue it via the regular Queue Prompt."
            )
        if not pod_template_id.strip():
            raise RuntimeError("pod_template_id is required.")
        if not network_volume_id.strip():
            raise RuntimeError("network_volume_id is required.")

        try:
            workflow = json.loads(_workflow_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Captured workflow is not valid JSON: {e}")
        if not isinstance(workflow, dict) or not workflow:
            raise RuntimeError(
                "Captured workflow is empty. Build a graph around this "
                "dispatcher node before clicking 'Run on RunPod'."
            )

        # Strip out this dispatcher node from the captured workflow so
        # the remote pod doesn't try to recurse into another dispatch.
        # The JS already does this; this is a belt-and-suspenders check.
        workflow = {
            nid: nd for nid, nd in workflow.items()
            if isinstance(nd, dict) and nd.get("class_type") != "RSRunOnRunPod"
        }
        if not workflow:
            raise RuntimeError(
                "Captured workflow contained only the dispatcher node. "
                "Connect or place at least one other node in the graph."
            )

        try:
            creds = resolve_creds(profile)
        except RunpodCredsError as e:
            raise RuntimeError(str(e))

        max_wait_seconds = max_wait_minutes * 60.0
        deadline = run_started_at + max_wait_seconds

        # ---- 2. Detect + plan asset uploads ----
        _phase(node_id, 1, 5, f"Resolved creds ({creds.label})")
        try:
            plan = workflow_assets.detect_and_rewrite(
                workflow,
                max_upload_bytes=runpod_client.MAX_UPLOAD_BYTES,
            )
        except workflow_assets.AssetWalkError as e:
            raise RuntimeError(str(e))

        n_assets = len(plan.assets)
        size_mb = plan.total_size_bytes / (1024 * 1024)
        _phase(
            node_id, 1, 5,
            f"Detected {n_assets} asset(s), {size_mb:.1f} MB to upload",
        )

        # ---- 3. Start (or reuse) the pod ----
        _phase(node_id, 2, 5, f"Starting pod ({gpu_type}, {creds.label})...")
        pod = runpod_client.start_pod(
            creds,
            template_id=pod_template_id.strip(),
            volume_id=network_volume_id.strip(),
            gpu_type=gpu_type,
            env={"RS_CAPTURE_CONSOLE": "1" if capture_console else "0"},
        )
        _phase(node_id, 2, 5, f"Pod {pod.pod_id} provisioned")

        pod_was_reused = (time.time() - pod.started_at) < 1.0  # heuristic
        # Track whether we should clean up on exit.
        should_stop_pod = bool(auto_stop)

        try:
            # ---- 4. Wait for ComfyUI to come up ----
            wait_budget = max(60.0, deadline - time.time())
            runpod_client.wait_for_comfy(
                pod.proxy_url,
                timeout_seconds=wait_budget,
                cancel_check=cancel_check,
                on_progress=lambda msg: _log(node_id, msg),
            )
            _phase(node_id, 2, 5, f"ComfyUI ready at {pod.proxy_url}")

            # ---- 5. Upload assets ----
            for i, asset in enumerate(plan.assets, start=1):
                cancel_check()
                _phase(
                    node_id, 3, 5,
                    f"Uploading {i}/{n_assets}: {asset.local_path.name} "
                    f"({asset.size_bytes / (1024 * 1024):.1f} MB)",
                )
                stored = runpod_client.upload_asset(
                    pod.proxy_url,
                    asset.local_path,
                    remote_name=asset.remote_name,
                )
                if stored != asset.remote_name:
                    # Pod stored under a different name (collision); track it.
                    plan.rewritten_workflow = _rebind_asset(
                        plan.rewritten_workflow,
                        asset, stored,
                    )

            if n_assets:
                _phase(node_id, 3, 5, f"Uploaded {n_assets} asset(s)")
            else:
                _phase(node_id, 3, 5, "No assets to upload")

            # ---- 6. Submit the workflow ----
            client_id = uuid.uuid4().hex
            _phase(node_id, 4, 5, "Submitting workflow...")
            prompt_id = runpod_client.submit_workflow(
                pod.proxy_url,
                plan.rewritten_workflow,
                client_id,
            )
            _log(node_id, f"Remote prompt_id: {prompt_id}")

            # ---- 7. Stream progress ----
            bridge = _RemoteProgressBridge(
                node_id, total_nodes=len(plan.rewritten_workflow),
            )
            try:
                history = runpod_client.stream_progress(
                    pod.proxy_url,
                    prompt_id,
                    client_id,
                    on_event=bridge,
                    cancel_check=cancel_check,
                    timeout_seconds=max(60.0, deadline - time.time()),
                )
            except KeyboardInterrupt:
                runpod_client.interrupt_remote(pod.proxy_url)
                raise
            except Exception:
                # Cancellation propagated up — make sure we tell the pod.
                runpod_client.interrupt_remote(pod.proxy_url)
                raise

            _phase(node_id, 4, 5, "Remote execution complete")

            # ---- 8. Download outputs ----
            output_dir = workflow_assets.output_dir_for_run(
                _comfy_output_dir(), prompt_id,
            )
            refs = runpod_client.collect_output_refs(history)
            saved: list[str] = []
            for ref in refs:
                cancel_check()
                local_dest = output_dir / ref["filename"]
                runpod_client.download_output(
                    pod.proxy_url,
                    filename=ref["filename"],
                    subfolder=ref["subfolder"],
                    asset_type=ref["type"],
                    local_dest=local_dest,
                )
                saved.append(str(local_dest))
                _log(node_id, f"Downloaded {ref['filename']}")

            _phase(
                node_id, 5, 5,
                f"Downloaded {len(saved)} output(s) to {output_dir}",
            )

            elapsed = time.time() - run_started_at
            summary = {
                "ok": True,
                "prompt_id": prompt_id,
                "pod_id": pod.pod_id,
                "pod_url": pod.proxy_url,
                "pod_label": creds.label,
                "pod_reused": pod_was_reused,
                "gpu_type": gpu_type,
                "elapsed_seconds": round(elapsed, 1),
                "asset_count": n_assets,
                "asset_bytes": plan.total_size_bytes,
                "output_dir": str(output_dir),
                "outputs": saved,
                "captured_console": capture_console,
            }
            return (json.dumps(summary, indent=2),)

        finally:
            if should_stop_pod:
                _phase(
                    node_id, 5, 5,
                    f"Stopping pod {pod.pod_id} (auto_stop=True)",
                )
                runpod_client.stop_pod(creds, pod.pod_id)
            else:
                _log(node_id, f"Leaving pod {pod.pod_id} running (auto_stop=False)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebind_asset(
    workflow: dict,
    asset: workflow_assets.AssetEntry,
    stored_name: str,
) -> dict:
    """Patch the workflow so a single asset's references use `stored_name`
    instead of the originally-planned remote_name. Used when ComfyUI
    munged the filename on collision."""
    out = dict(workflow)
    node = out.get(asset.node_id)
    if not isinstance(node, dict):
        return out
    new_inputs = dict(node.get("inputs") or {})
    if new_inputs.get(asset.input_name) == asset.remote_name:
        new_inputs[asset.input_name] = stored_name
    new_node = dict(node)
    new_node["inputs"] = new_inputs
    out[asset.node_id] = new_node
    return out
