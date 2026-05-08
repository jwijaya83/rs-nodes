# RunPod remote dispatch — operator setup

This is the one-time setup needed before the **RS Run on RunPod**
dispatcher node will work. After this is done, day-to-day use is just:
drop the dispatcher node into a graph, click **Run on RunPod**, watch
outputs appear in `output/runpod/<prompt_id>/`.

## What gets dispatched

The dispatcher node captures the rest of your local graph, sends it to
a RunPod pod that mounts your network volume, runs it there, downloads
the outputs back to your local `output/runpod/<prompt_id>/`, and stops
the pod. Heavy nodes that would OOM your local 16 GB GPU run on a 24+
GB cloud GPU instead — no other change to how you author workflows.

## One-time setup

### 1. Create a RunPod network volume

[RunPod console → Storage](https://www.runpod.io/console/user/storage) →
**New Network Volume**.

Recommended starting size: **100 GB**. The volume is resizable, so
start small and grow as needed.

Choose a region close to wherever you usually work (lower latency for
the upload/download steps). Pick the same region every time you start
a pod against this volume.

Note the **volume ID** — you'll paste it into the dispatcher node's
`network_volume_id` widget.

### 2. Provision the volume

You need to put two things on the volume:

1. The **startup script** at `/workspace/startup.sh`.
2. Optionally, models you intend to use. (The dispatcher's asset
   uploader will move per-run inputs over for you, but bulk models like
   `ltx-2.3-22b-dev-fp8.safetensors` should live on the volume so the
   pod doesn't redownload them every cold start.)

Easiest path:

1. Start a temporary pod with the volume attached. Use any container
   with shell + git (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4`).
2. SSH or use the web terminal into the pod.
3. Place `startup.sh` on the volume:

       cd /workspace
       git clone https://github.com/richservo/rs-nodes.git tmp-rs
       cp tmp-rs/runpod/startup.sh /workspace/startup.sh
       chmod +x /workspace/startup.sh
       rm -rf tmp-rs

4. Place your models under `/workspace/ComfyUI/models/...` matching the
   layout your workflows expect:

       /workspace/ComfyUI/models/checkpoints/<your model>.safetensors
       /workspace/ComfyUI/models/vae/<your vae>.safetensors
       /workspace/ComfyUI/models/loras/<your loras>...
       /workspace/ComfyUI/models/text_encoders/...

   The pod's ComfyUI install (cloned by `startup.sh`) will share this
   layout, so workflows that work locally with these filenames work
   remotely without translation.

5. Stop the temporary pod. The volume keeps everything you placed.

### 3. Create a pod template

[RunPod console → Templates](https://www.runpod.io/console/user/templates) →
**New Template**.

- **Container image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4` (or any
  pytorch + CUDA image whose torch version matches your local one).
- **Container Disk**: 50 GB is plenty (everything persistent goes on
  the network volume).
- **Volume Mount Path**: `/workspace`
- **Container Start Command**:

      bash /workspace/startup.sh

- **Expose HTTP Ports**: `8188`
- **Environment Variables**:
  - `RS_INSTALL_OLLAMA=0` (set to `1` only if you'll caption datasets
    on the pod — not needed for inference workflows).
  - `RS_PREFETCH_INSIGHTFACE=0` (set to `1` only if your workflow uses
    InsightFace face detection on the pod).

Note the **template ID** — you'll paste it into the dispatcher's
`pod_template_id` widget.

### 4. Set up local credentials

Copy `runpod/credentials.ini.example` to:

- Windows: `%USERPROFILE%\.runpod\credentials.ini`
- POSIX: `~/.runpod/credentials.ini`

Edit it with your real `api_key` + `user_id` from
[RunPod settings](https://www.runpod.io/console/user/settings).

Multiple accounts: add a section per account (e.g. `[personal]`,
`[work]`) and select via the dispatcher's `profile` widget.

Single-account first run: if you don't want a credentials file yet,
set `RUNPOD_API_KEY` and `RUNPOD_USER_ID` env vars instead — the
dispatcher uses them as a fallback.

The API key needs the following permissions:
- `Pods: Create / Read / Stop` (start a pod, list to find reusable
  ones, stop after the run).
- `NetworkVolumes: Read` (resolve the volume by ID).

### 5. (Optional) Install `websocket-client` locally

The dispatcher streams per-step progress (KSampler step count, etc.)
back to the local UI via WebSocket. The Python side of that needs
`websocket-client`:

    pip install websocket-client

Without it, the dispatcher falls back to polling `/history`, which
still works but doesn't give you a live progress bar — you only see
node-completion events.

## Day-to-day use

1. Build your workflow in ComfyUI as normal.
2. Drop in **RS Run on RunPod**. Set:
   - `pod_template_id` — from step 3 above.
   - `network_volume_id` — from step 1.
   - `gpu_type` — pick from the dropdown.
   - `auto_stop` — leave **on** unless you're iterating fast and want
     the pod to stay warm between dispatches.
3. Click **Run on RunPod** on that node. **Don't** click the global
   Queue Prompt button — that would try to run the heavy nodes
   locally.
4. Watch the status panel on the node. Outputs land in
   `output/runpod/<prompt_id>/`.

## Known limits in v1

- **Asset upload cap: 2 GB per file.** Anything larger must be staged
  on the network volume directly (training datasets, big LoRAs, etc.).
  A bulk-upload tool that rsyncs to the volume is a planned follow-up.
- **Directory inputs are rejected.** Nodes like
  `RSLTXVPrepareDataset.media_folder` that take a folder path will
  surface a clear error pointing at the bulk-upload follow-up — stage
  those datasets on the network volume.
- **Model paths are not translated.** Whatever model filenames your
  workflow references must exist on the network volume under the same
  paths. Per-account name remapping is a v2 feature.
- **One pod per dispatch.** No multi-pod fan-out yet.

## Troubleshooting

- **"Pod did not respond at <url>/system_stats within 600s"** —
  pod boot took longer than the timeout. Bump `max_wait_minutes` on
  the dispatcher; first-time pulls of large models can take a while.
- **"No RunPod credentials found"** — see step 4 above.
- **"Profile 'X' not found"** — the dispatcher's `profile` widget
  doesn't match any section in `credentials.ini`.
- **Outputs missing** — check the status panel for download errors.
  Each output is downloaded immediately after the remote run, so any
  failure shows up there.
- **Pod still running after a crash** — the dispatcher tries to stop
  the pod even on errors, but if the local Python process was killed
  outright, the pod may linger. Stop it manually from the RunPod
  console.
