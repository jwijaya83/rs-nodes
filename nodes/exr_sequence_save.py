"""EXR image-sequence save node for HDR / log workflows.

Writes each frame of an IMAGE tensor as a separate OpenEXR file into an
auto-incrementing directory under the ComfyUI output folder. Float32
pixel values are preserved (no clamping), so HDR / LogC content keeps
its full dynamic range. Daisy-chain compatible: passes images and audio
through so a ProRes/MP4 saver can follow the same graph.
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
        import cv2
        import numpy as np
        import folder_paths
        from ..utils import runpod_staging

        # OpenCV needs the OPENCV_IO_ENABLE_OPENEXR flag on some builds —
        # set it before writing in case the calling process hasn't.
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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

        # Compression + pixel type constants, guarded so we fall back to
        # sane defaults if the OpenCV build doesn't expose them.
        def _cv2const(name, default):
            return getattr(cv2, name, default)

        compression_map = {
            "none": _cv2const("IMWRITE_EXR_COMPRESSION_NO", 0),
            "rle": _cv2const("IMWRITE_EXR_COMPRESSION_RLE", 1),
            "zips": _cv2const("IMWRITE_EXR_COMPRESSION_ZIPS", 2),
            "zip": _cv2const("IMWRITE_EXR_COMPRESSION_ZIP", 3),
            "piz": _cv2const("IMWRITE_EXR_COMPRESSION_PIZ", 4),
            "pxr24": _cv2const("IMWRITE_EXR_COMPRESSION_PXR24", 5),
        }
        cv2_compression = compression_map.get(compression, compression_map["zip"])
        exr_type = (
            _cv2const("IMWRITE_EXR_TYPE_HALF", 1) if bit_depth == "16"
            else _cv2const("IMWRITE_EXR_TYPE_FLOAT", 2)
        )
        type_key = _cv2const("IMWRITE_EXR_TYPE", 48)
        compression_key = _cv2const("IMWRITE_EXR_COMPRESSION", 49)

        # Per-frame write. Pixel values are NOT clamped — HDR / LogC
        # content above 1.0 is preserved exactly as it came off the model.
        num_frames = images.shape[0]
        has_alpha = images.shape[-1] == 4
        padding = max(3, int(frame_padding))
        wrote = 0
        for i in range(num_frames):
            frame = images[i].to(device="cpu", dtype=torch.float32).numpy()
            # Tensor is RGB(A); OpenCV writes BGR(A). Reverse the colour
            # axis in place with a view, then .copy() so imwrite gets a
            # contiguous buffer.
            if has_alpha:
                frame = frame[..., [2, 1, 0, 3]].copy()
            else:
                frame = frame[..., [2, 1, 0]].copy()

            frame_path = os.path.join(out_dir, f"{name_base}_{i:0{padding}d}.exr")
            params = [
                int(compression_key), int(cv2_compression),
                int(type_key), int(exr_type),
            ]
            ok = cv2.imwrite(frame_path, frame, params)
            if not ok:
                raise RuntimeError(
                    f"cv2.imwrite failed for {frame_path}. OpenCV may have "
                    f"been built without OpenEXR support. Install "
                    f"opencv-python (not the headless variant) or use "
                    f"`pip install opencv-contrib-python`."
                )
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
