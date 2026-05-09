"""RunPod + ComfyUI client.

Two API surfaces wrapped here:

1. RunPod REST API (https://rest.runpod.io/v1) for pod lifecycle:
   start, list, stop. Authenticated per-call with the resolved
   RunpodCreds.

2. The remote pod's ComfyUI REST + WebSocket once it's up:
   /system_stats, /prompt, /history, /view, /upload/image, /interrupt,
   /ws — used for asset upload, workflow submission, progress
   streaming, output download.

All calls are synchronous + cancel-aware. A `cancel_check` callable is
threaded through the long-running ones so ComfyUI's interrupt button
can break out of polls and the dispatcher can stop the pod cleanly
instead of leaving it billing.

The pod proxy URL pattern is RunPod's standard:
    https://{pod_id}-{port}.proxy.runpod.net
We expose that as `proxy_url(pod_id, port)` so callers don't bake the
format in.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from .runpod_credentials import RunpodCreds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants + types
# ---------------------------------------------------------------------------

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
COMFY_PORT = 8188

# How long to wait between polls for things-that-take-a-while.
POLL_INTERVAL_SECONDS = 2.0

# Largest file size we'll auto-upload via ComfyUI's /upload/image.
# Anything bigger (training datasets) goes through the deferred bulk
# uploader (see plan: "Bulk dataset upload to network volume").
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class RunpodApiError(RuntimeError):
    """Anything that went wrong talking to RunPod's API."""


class RemoteComfyError(RuntimeError):
    """Anything that went wrong talking to ComfyUI on the pod."""


@dataclass
class PodHandle:
    """Describes a running pod we'll dispatch to."""
    pod_id: str
    proxy_url: str
    gpu_type: str
    started_at: float


# ---------------------------------------------------------------------------
# RunPod REST helpers
# ---------------------------------------------------------------------------

def _runpod_headers(creds: RunpodCreds) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {creds.api_key}",
        "Content-Type": "application/json",
    }


def _runpod_request(
    method: str,
    path: str,
    creds: RunpodCreds,
    *,
    json_body: dict | None = None,
    timeout: float = 30.0,
) -> Any:
    """Thin wrapper around RunPod's REST API. Raises RunpodApiError on
    non-2xx responses with whatever the server told us about why."""
    url = f"{RUNPOD_REST_BASE}{path}"
    try:
        resp = requests.request(
            method, url,
            headers=_runpod_headers(creds),
            json=json_body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RunpodApiError(f"{method} {url}: network error: {e}") from e

    if not resp.ok:
        body = resp.text[:500]
        raise RunpodApiError(
            f"{method} {url}: HTTP {resp.status_code} {resp.reason}\n  {body}"
        )
    if not resp.content:
        return None
    try:
        return resp.json()
    except ValueError:
        return resp.text


def proxy_url(pod_id: str, port: int = COMFY_PORT) -> str:
    """RunPod's standard port-proxy URL pattern."""
    return f"https://{pod_id}-{port}.proxy.runpod.net"


# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------

def list_pods(creds: RunpodCreds) -> list[dict]:
    """Return all pods on this account regardless of status."""
    data = _runpod_request("GET", "/pods", creds)
    if isinstance(data, dict):
        # API has historically returned either a bare list or an
        # envelope; tolerate both.
        return data.get("data") or data.get("pods") or []
    return list(data) if isinstance(data, list) else []


def find_reusable_pod(
    creds: RunpodCreds,
    template_id: str,
    volume_id: str,
) -> dict | None:
    """Find an already-running pod with the same template + volume so a
    rapid second dispatch reuses it instead of cold-starting a new one.
    Returns the pod dict or None."""
    for pod in list_pods(creds):
        status = (pod.get("desiredStatus") or pod.get("status") or "").upper()
        if status not in ("RUNNING",):
            continue
        if pod.get("templateId") != template_id:
            continue
        # Volume comes back under a few different keys depending on API
        # version. Match any of them.
        pod_volume = (
            pod.get("networkVolumeId")
            or pod.get("volumeId")
            or (pod.get("networkVolume") or {}).get("id")
        )
        if pod_volume != volume_id:
            continue
        return pod
    return None


def start_pod(
    creds: RunpodCreds,
    *,
    template_id: str,
    volume_id: str,
    gpu_type: str,
    name: str | None = None,
    env: dict | None = None,
    container_disk_gb: int = 50,
) -> PodHandle:
    """Start a pod (or reuse an already-running matching one).

    Returns a PodHandle once the pod has reached RUNNING status — note
    this is RunPod's "the container has started," NOT "ComfyUI is
    serving." Callers should follow up with wait_for_comfy().
    """
    existing = find_reusable_pod(creds, template_id, volume_id)
    if existing is not None:
        pod_id = existing.get("id") or existing.get("podId")
        logger.info(f"Reusing already-running pod {pod_id} ({creds.label})")
        return PodHandle(
            pod_id=pod_id,
            proxy_url=proxy_url(pod_id),
            gpu_type=existing.get("gpuTypeId", gpu_type),
            started_at=time.time(),
        )

    body = {
        "name": name or f"rs-nodes-{uuid.uuid4().hex[:8]}",
        "templateId": template_id,
        "gpuTypeIds": [gpu_type],
        "networkVolumeId": volume_id,
        "containerDiskInGb": container_disk_gb,
        "ports": f"{COMFY_PORT}/http",
        "env": env or {},
    }
    logger.info(
        f"Starting pod ({creds.label}, gpu={gpu_type}, "
        f"template={template_id}, volume={volume_id})"
    )
    resp = _runpod_request("POST", "/pods", creds, json_body=body)

    pod_id = resp.get("id") if isinstance(resp, dict) else None
    if not pod_id:
        raise RunpodApiError(
            f"Pod start did not return an id. Response: {resp}"
        )

    return PodHandle(
        pod_id=pod_id,
        proxy_url=proxy_url(pod_id),
        gpu_type=gpu_type,
        started_at=time.time(),
    )


def stop_pod(creds: RunpodCreds, pod_id: str) -> None:
    """Stop the pod (preserves the network volume). Idempotent."""
    try:
        _runpod_request("POST", f"/pods/{pod_id}/stop", creds)
    except RunpodApiError as e:
        logger.warning(f"stop_pod({pod_id}) failed: {e}")


# ---------------------------------------------------------------------------
# Remote ComfyUI: readiness, asset upload, workflow submit
# ---------------------------------------------------------------------------

def wait_for_comfy(
    pod_url: str,
    timeout_seconds: float = 600.0,
    cancel_check: Callable[[], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Poll <pod_url>/system_stats until 200 OK or timeout.

    Raises RemoteComfyError on timeout. Cooperates with cancel_check.
    """
    deadline = time.time() + timeout_seconds
    last_log = 0.0
    while time.time() < deadline:
        if cancel_check is not None:
            cancel_check()
        try:
            r = requests.get(f"{pod_url}/system_stats", timeout=10.0)
            if r.ok:
                if on_progress:
                    on_progress("Pod ready")
                return
        except requests.RequestException:
            pass

        # Log every 30s so the user knows we're still waiting.
        now = time.time()
        if now - last_log > 30:
            elapsed = int(now - (deadline - timeout_seconds))
            if on_progress:
                on_progress(f"Waiting for ComfyUI ({elapsed}s)")
            last_log = now
        time.sleep(POLL_INTERVAL_SECONDS)

    raise RemoteComfyError(
        f"ComfyUI did not respond at {pod_url}/system_stats within "
        f"{int(timeout_seconds)}s"
    )


def upload_asset(
    pod_url: str,
    local_path: Path,
    *,
    remote_name: str | None = None,
    asset_type: str = "input",
    timeout_seconds: float = 600.0,
) -> str:
    """Upload a single file to ComfyUI's /upload/image endpoint.

    Despite its name, /upload/image accepts any binary content (images,
    videos, audio). The endpoint stores it under ComfyUI's input/ dir
    by default; remote workflows reference it by its returned filename.

    Args:
        pod_url: The remote ComfyUI URL.
        local_path: Local file to upload.
        remote_name: Override the filename on the pod side. If None,
            uses local_path.name.
        asset_type: Comfy's `type` form field — "input" (default),
            "temp", or "output".
        timeout_seconds: Per-request timeout (uploads can be slow).

    Returns:
        The filename ComfyUI stored the file under.
    """
    size = local_path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise RemoteComfyError(
            f"{local_path} is {size / 1024**3:.1f} GB, exceeds "
            f"{MAX_UPLOAD_BYTES / 1024**3:.0f} GB inline-upload limit. "
            "Big files (training datasets etc.) must be staged on the "
            "network volume directly. See: deferred bulk-upload tool."
        )

    name = remote_name or local_path.name
    with local_path.open("rb") as fp:
        files = {"image": (name, fp, "application/octet-stream")}
        data = {"overwrite": "true", "type": asset_type}
        try:
            r = requests.post(
                f"{pod_url}/upload/image",
                files=files,
                data=data,
                timeout=timeout_seconds,
            )
        except requests.RequestException as e:
            raise RemoteComfyError(
                f"upload {local_path.name} failed: {e}"
            ) from e

    if not r.ok:
        raise RemoteComfyError(
            f"upload {local_path.name} HTTP {r.status_code}: {r.text[:300]}"
        )

    body = r.json() if r.content else {}
    # ComfyUI returns {"name": stored_name, "subfolder": "", "type": ...}
    return body.get("name", name)


def submit_workflow(
    pod_url: str,
    workflow_prompt: dict,
    client_id: str,
    *,
    extra_data: dict | None = None,
) -> str:
    """POST /prompt and return the prompt_id."""
    body = {
        "prompt": workflow_prompt,
        "client_id": client_id,
    }
    if extra_data:
        body["extra_data"] = extra_data

    try:
        r = requests.post(f"{pod_url}/prompt", json=body, timeout=60.0)
    except requests.RequestException as e:
        raise RemoteComfyError(f"submit_workflow failed: {e}") from e

    if not r.ok:
        raise RemoteComfyError(
            f"submit_workflow HTTP {r.status_code}: {r.text[:500]}"
        )

    pid = r.json().get("prompt_id")
    if not pid:
        raise RemoteComfyError(f"submit_workflow returned no prompt_id: {r.text}")
    return pid


def fetch_history(pod_url: str, prompt_id: str) -> dict | None:
    """GET /history/{prompt_id}. Returns the per-prompt history dict
    (keys: prompt, outputs, status) or None if not yet recorded."""
    try:
        r = requests.get(f"{pod_url}/history/{prompt_id}", timeout=15.0)
    except requests.RequestException as e:
        raise RemoteComfyError(f"fetch_history failed: {e}") from e
    if not r.ok:
        raise RemoteComfyError(
            f"fetch_history HTTP {r.status_code}: {r.text[:300]}"
        )
    body = r.json() or {}
    return body.get(prompt_id)


def interrupt_remote(pod_url: str) -> None:
    """POST /interrupt to halt the currently-executing prompt."""
    try:
        requests.post(f"{pod_url}/interrupt", timeout=10.0)
    except requests.RequestException as e:
        logger.warning(f"interrupt_remote({pod_url}) failed: {e}")


# ---------------------------------------------------------------------------
# Output download
# ---------------------------------------------------------------------------

def download_output(
    pod_url: str,
    filename: str,
    subfolder: str,
    asset_type: str,
    local_dest: Path,
    timeout_seconds: float = 600.0,
) -> Path:
    """Stream a single output file from the pod to a local path.

    `filename`, `subfolder`, `asset_type` come from the history dict's
    outputs entry (one entry per node-output). Streams to a temp file
    then renames so a partial download can't masquerade as complete.
    """
    params = {"filename": filename, "subfolder": subfolder, "type": asset_type}
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_dest.with_suffix(local_dest.suffix + ".tmp")
    try:
        with requests.get(
            f"{pod_url}/view",
            params=params,
            stream=True,
            timeout=timeout_seconds,
        ) as r:
            r.raise_for_status()
            with tmp.open("wb") as fp:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fp.write(chunk)
    except requests.RequestException as e:
        tmp.unlink(missing_ok=True)
        raise RemoteComfyError(
            f"download_output({filename}) failed: {e}"
        ) from e

    tmp.replace(local_dest)
    return local_dest


def collect_output_refs(history: dict) -> list[dict]:
    """Walk a prompt's history dict to flatten all output file references.

    Returns a list of dicts with keys: filename, subfolder, type, node_id.
    Same shape ComfyUI's /view expects.
    """
    refs: list[dict] = []
    outputs = history.get("outputs", {}) if isinstance(history, dict) else {}
    for node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        # Common output keys: "images", "gifs", "videos", "audio", "clips".
        # We accept any list of dicts that has filename + subfolder.
        for key, val in node_out.items():
            if not isinstance(val, list):
                continue
            for entry in val:
                if not isinstance(entry, dict):
                    continue
                if "filename" not in entry:
                    continue
                refs.append({
                    "filename": entry["filename"],
                    "subfolder": entry.get("subfolder", ""),
                    "type": entry.get("type", "output"),
                    "node_id": node_id,
                    "category": key,
                })
    return refs


# ---------------------------------------------------------------------------
# Progress streaming
# ---------------------------------------------------------------------------

def stream_progress(
    pod_url: str,
    prompt_id: str,
    client_id: str,
    *,
    on_event: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    timeout_seconds: float = 6 * 3600.0,
) -> dict:
    """Wait for the remote prompt to complete, forwarding progress
    events along the way. Returns the final history dict.

    Tries the WebSocket if `websocket-client` is installed (smooth
    per-step progress), otherwise polls /history (works but no
    in-step progress bar).

    Raises:
        RemoteComfyError: timeout or remote execution error
    """
    # Try the WebSocket path first — gives per-step progress events.
    try:
        return _stream_progress_ws(
            pod_url, prompt_id, client_id,
            on_event=on_event,
            cancel_check=cancel_check,
            timeout_seconds=timeout_seconds,
        )
    except _WebSocketUnavailable:
        logger.info(
            "websocket-client not installed; falling back to /history "
            "polling for remote progress (no per-step progress bar)"
        )
    return _stream_progress_poll(
        pod_url, prompt_id,
        on_event=on_event,
        cancel_check=cancel_check,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


class _WebSocketUnavailable(RuntimeError):
    pass


def _stream_progress_ws(
    pod_url: str,
    prompt_id: str,
    client_id: str,
    *,
    on_event: Callable[[dict], None] | None,
    cancel_check: Callable[[], None] | None,
    timeout_seconds: float,
) -> dict:
    """WebSocket-based progress relay. Raises _WebSocketUnavailable if
    the websocket-client lib isn't installed."""
    try:
        import websocket  # websocket-client
    except ImportError as e:
        raise _WebSocketUnavailable() from e

    ws_url = pod_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws?clientId={client_id}"

    ws = websocket.WebSocket()
    ws.settimeout(5.0)  # short per-recv so cancel_check can run between
    try:
        ws.connect(ws_url)
    except Exception as e:
        raise RemoteComfyError(f"WebSocket connect to {ws_url} failed: {e}") from e

    deadline = time.time() + timeout_seconds
    finished = False
    try:
        while not finished and time.time() < deadline:
            if cancel_check is not None:
                try:
                    cancel_check()
                except Exception:
                    interrupt_remote(pod_url)
                    raise

            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                logger.warning(f"WebSocket recv error: {e}; falling back to polling")
                break

            if not msg:
                continue
            if isinstance(msg, bytes):
                # Binary previews — skip
                continue
            try:
                event = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if on_event is not None:
                try:
                    on_event(event)
                except Exception as cb_err:
                    logger.warning(f"on_event callback raised: {cb_err}")

            etype = event.get("type")
            edata = event.get("data") or {}
            # Completion signal: executing event with node=null AND
            # matching prompt_id means the queue item finished.
            if (
                etype == "executing"
                and edata.get("node") is None
                and edata.get("prompt_id") == prompt_id
            ):
                finished = True
                break
            if etype in ("execution_error", "execution_interrupted"):
                if edata.get("prompt_id") == prompt_id:
                    raise RemoteComfyError(
                        f"Remote execution {etype}: {edata}"
                    )
    finally:
        try:
            ws.close()
        except Exception:
            pass

    if not finished:
        raise RemoteComfyError(
            f"Remote execution did not complete within {int(timeout_seconds)}s"
        )

    history = fetch_history(pod_url, prompt_id)
    if history is None:
        raise RemoteComfyError(
            f"WebSocket reported completion but /history/{prompt_id} is empty"
        )
    return history


def _stream_progress_poll(
    pod_url: str,
    prompt_id: str,
    *,
    on_event: Callable[[dict], None] | None,
    cancel_check: Callable[[], None] | None,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict:
    """Polling fallback. No per-step progress, but reports
    node-completion events as they appear in /history."""
    deadline = time.time() + timeout_seconds
    seen_outputs: set[str] = set()
    while time.time() < deadline:
        if cancel_check is not None:
            try:
                cancel_check()
            except Exception:
                interrupt_remote(pod_url)
                raise

        history = fetch_history(pod_url, prompt_id)
        if history is not None:
            # Synthesize "executed" events for nodes whose outputs we've
            # newly observed, so the on_event callback can update the
            # local UI even without WebSocket.
            outputs = history.get("outputs", {})
            for node_id in outputs:
                if node_id not in seen_outputs:
                    seen_outputs.add(node_id)
                    if on_event is not None:
                        try:
                            on_event({
                                "type": "executed",
                                "data": {
                                    "node": node_id,
                                    "prompt_id": prompt_id,
                                    "output": outputs[node_id],
                                },
                            })
                        except Exception:
                            pass

            status = history.get("status", {})
            if status.get("completed"):
                return history
            if status.get("status_str") == "error":
                raise RemoteComfyError(
                    f"Remote execution errored: {status}"
                )

        time.sleep(poll_interval_seconds)

    raise RemoteComfyError(
        f"Remote execution did not complete within {int(timeout_seconds)}s"
    )


# ---------------------------------------------------------------------------
# Asset hash helper (used by workflow_assets.py)
# ---------------------------------------------------------------------------

def file_sha256_short(path: Path, length: int = 12) -> str:
    """Compute SHA256 of a file, return the first `length` hex chars.
    Used to build cache-friendly remote names so re-dispatches reuse
    uploads when the source file hasn't changed."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]
