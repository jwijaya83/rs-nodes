"""EXR image-sequence save node for HDR / log workflows.

Writes each frame of an IMAGE tensor as a separate OpenEXR file into an
auto-incrementing directory under the ComfyUI output folder. Float32
pixel values are preserved (no clamping), so HDR / LogC content keeps
its full dynamic range. Daisy-chain compatible: passes images and audio
through so a ProRes/MP4 saver can follow the same graph.

Backend: official OpenEXR Python bindings (`pip install OpenEXR Imath`).
Uses those rather than cv2 because opencv-python's prebuilt wheels ship
with the EXR codec disabled, and there's no way to enable it after cv2
is imported (the codec state is baked in at C-extension load time).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


class RSEXRSequenceSave:
    """Save video frames as an EXR image sequence in a new directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Passed through — EXR sequences don't contain audio, but this lets a ProRes/MP4 saver downstream still receive it."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
                                  "tooltip": "Directory index suffix. 0 = auto-increment (same pattern as the ProRes save node)."}),
                "output_dir": ("STRING", {"default": "", "tooltip":
                    "When runpod is OFF: subdirectory under ComfyUI output (the EXR sequence "
                    "goes into a new subdirectory below this). When runpod is ON: absolute LOCAL "
                    "path on your home machine where the pull tool will deposit the sequence "
                    "directory (e.g. E:\\Comfy\\Source\\Project\\renders\\cut01)."}),
                "runpod": ("BOOLEAN", {"default": False, "tooltip":
                    "Stage this sequence for later pull from a RunPod pod. EXR frames are written "
                    "to <output>/runpod_pending/<run_id>/<seqname>/ on the pod with a manifest, "
                    "and a separate local-side pull tool moves the whole directory to `output_dir`. "
                    "Off = save directly to output_dir as today."}),
                "pull_server_url": ("STRING", {"default": "http://localhost:8765/pull", "tooltip":
                    "Only used when runpod is ON. URL of the local pull server (reached on the "
                    "pod via SSH reverse tunnel). Empty = skip the auto-pull; you'd run a manual "
                    "pull from the manifests later."}),
                "bit_depth": (["16", "32"], {"default": "16",
                                              "tooltip": "EXR float precision. 16 = half-float (smaller files, plenty of range for display / grading). 32 = full float (max precision, ~2x file size)."}),
                "compression": (["zip", "zips", "piz", "rle", "pxr24", "none"], {"default": "zip",
                                                                                   "tooltip": "EXR compression. zip = lossless, good general default. piz = lossless, best for grainy content. pxr24 = lossy 24-bit float (not recommended for linear HDR). none = uncompressed, fastest write."}),
                "frame_padding": ("INT", {"default": 6, "min": 3, "max": 10, "step": 1,
                                          "tooltip": "Zero-pad width for per-frame filenames. 6 supports up to 999,999 frames per sequence."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "dir_path")
    FUNCTION = "save_exr_sequence"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save_exr_sequence(self, images, filename_prefix,
                          audio=None, index=0, output_dir="",
                          bit_depth="16", compression="zip",
                          frame_padding=6, runpod=False,
                          pull_server_url="http://localhost:8765/pull"):
        import numpy as np
        import folder_paths
        from ..utils import runpod_staging

        # Use the official OpenEXR Python bindings instead of cv2.
        # cv2's prebuilt wheels (opencv-python 4.13.0 on Windows in
        # particular) ship with the OpenEXR codec disabled at build
        # time — you get a runtime "OpenEXR codec is disabled" error
        # that no env var can fix once cv2 is loaded. OpenEXR is the
        # canonical pure-EXR library, has prebuilt wheels for every
        # major platform, and doesn't care about cv2's state.
        try:
            import OpenEXR
            import Imath
        except ImportError as e:
            raise RuntimeError(
                "RSEXRSequenceSave requires the OpenEXR Python lib. Install it in "
                "your ComfyUI venv:  pip install OpenEXR Imath\n"
                f"(Underlying error: {e})"
            ) from e

        # Resolve base output path. runpod=True overrides to a per-run
        # staging directory; the user-specified output_dir is captured
        # in the manifest for the local-side pull tool.
        staging_handle = None
        if runpod:
            target_dir = output_dir.strip()
            if not target_dir:
                raise ValueError(
                    "RSEXRSequenceSave: runpod=True requires output_dir to be "
                    "set to the local path where the pull tool should deposit "
                    "the sequence."
                )
            staging_handle = runpod_staging.allocate(prefix_hint=filename_prefix)
            base = staging_handle.staging_dir
        else:
            base = folder_paths.get_output_directory()
            if output_dir.strip():
                base = os.path.join(base, output_dir.strip())
        os.makedirs(base, exist_ok=True)

        # Split prefix into subdirectory + directory-name base so the same
        # prefix conventions as the ProRes saver work (e.g. "cut01/takeA").
        if os.sep in filename_prefix or "/" in filename_prefix:
            sub_dir, name_base = os.path.split(filename_prefix)
            base = os.path.join(base, sub_dir)
            os.makedirs(base, exist_ok=True)
        else:
            name_base = filename_prefix

        # Resolve the per-sequence directory. Matches ProRes's file-index
        # pattern — `{name}_00001`, `{name}_00002`, ...
        if index > 0:
            dirname = f"{name_base}_{index:05d}"
        else:
            counter = 0
            try:
                for d in os.listdir(base):
                    full = os.path.join(base, d)
                    if not os.path.isdir(full):
                        continue
                    if not d.startswith(name_base + "_"):
                        continue
                    try:
                        num = int(d[len(name_base) + 1:])
                        counter = max(counter, num)
                    except ValueError:
                        pass
            except FileNotFoundError:
                pass
            dirname = f"{name_base}_{counter + 1:05d}"

        out_dir = os.path.join(base, dirname)
        os.makedirs(out_dir, exist_ok=True)

        # OpenEXR compression map. Strings line up with the node's UI
        # dropdown; falls back to ZIP if the build doesn't have a
        # particular variant (unlikely for the standard set).
        comp_map = {
            "none": Imath.Compression.NO_COMPRESSION,
            "rle": Imath.Compression.RLE_COMPRESSION,
            "zips": Imath.Compression.ZIPS_COMPRESSION,
            "zip": Imath.Compression.ZIP_COMPRESSION,
            "piz": Imath.Compression.PIZ_COMPRESSION,
            "pxr24": Imath.Compression.PXR24_COMPRESSION,
        }
        exr_compression = Imath.Compression(comp_map.get(compression, comp_map["zip"]))
        # half (16) = Imath.PixelType.HALF, full (32) = FLOAT.
        pixel_type = Imath.PixelType(
            Imath.PixelType.HALF if bit_depth == "16" else Imath.PixelType.FLOAT
        )
        # numpy dtype matching the EXR per-pixel storage; we encode at
        # that precision so the writer doesn't have to convert.
        np_dtype = np.float16 if bit_depth == "16" else np.float32

        # Per-frame write. Pixel values are NOT clamped — HDR / LogC
        # content above 1.0 is preserved exactly as it came off the model.
        num_frames = images.shape[0]
        has_alpha = images.shape[-1] == 4
        padding = max(3, int(frame_padding))
        wrote = 0
        # Channel set (DataWindow + Channels) is the same for every frame
        # in a sequence, so build the header template once.
        for i in range(num_frames):
            frame = images[i].to(device="cpu", dtype=torch.float32).numpy()
            # Tensor is RGB(A) in [B,H,W,C]. OpenEXR wants per-channel
            # contiguous byte strings — slice each channel, cast to the
            # target precision, then call .tobytes().
            h, w = frame.shape[:2]
            header = OpenEXR.Header(w, h)
            header["compression"] = exr_compression
            channels = ["R", "G", "B"] + (["A"] if has_alpha else [])
            header["channels"] = {ch: Imath.Channel(pixel_type) for ch in channels}

            channel_data = {}
            for ch_idx, ch_name in enumerate(channels):
                arr = np.ascontiguousarray(frame[..., ch_idx].astype(np_dtype))
                channel_data[ch_name] = arr.tobytes()

            frame_path = os.path.join(out_dir, f"{name_base}_{i:0{padding}d}.exr")
            try:
                out = OpenEXR.OutputFile(frame_path, header)
                try:
                    out.writePixels(channel_data)
                finally:
                    out.close()
            except Exception as e:
                raise RuntimeError(
                    f"OpenEXR write failed for {frame_path}: {e}"
                ) from e
            wrote += 1

        logger.info(
            f"Saved EXR sequence ({wrote} frames, {bit_depth}-bit {compression}, "
            f"alpha={has_alpha}): {out_dir}"
        )

        # Mark the staging dir ready for the local pull tool. Done LAST
        # so a partially-written sequence can't be picked up.
        if staging_handle is not None:
            target_dir_str = output_dir.strip()
            target_path = os.path.join(target_dir_str, os.path.basename(out_dir))
            runpod_staging.write_manifest(
                staging_handle,
                target_path=target_path,
                kind="directory",
                saver="RSEXRSequenceSave",
                extra={"frames": wrote, "bit_depth": bit_depth,
                       "compression": compression},
            )
            # Best-effort callback so the local pull server scps the
            # sequence home immediately. Failures are non-fatal — the
            # manifest stays on the pod for a later manual pull.
            runpod_staging.notify_pull(
                staging_handle,
                target_path=target_path,
                kind="directory",
                callback_url=pull_server_url,
            )

        return (images, audio, out_dir)
