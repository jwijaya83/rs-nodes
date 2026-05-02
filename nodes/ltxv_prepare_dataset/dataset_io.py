"""Dataset I/O helpers for the prepare-dataset node.

Owns the dataset.json layer: entry normalization, path derivation
for encoded artifacts, audit/repair against on-disk state, and clean
rejection of bad entries. dataset.json is the source of truth ONLY
for encoding work; once artifacts exist on disk they take precedence.
"""

import json
import logging
import shutil
import time as _t
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# COND_TOKEN_LIMIT default for audit's bad-condition detection. Lives in
# encoding.py too (the encoder pads to this); here it's the audit's
# threshold for blanking captions whose condition is over-limit.
COND_TOKEN_LIMIT = 128


def normalize_loaded_entries(entries, output_dir):
    """Resolve every entry's media_path to an absolute path in-memory so
    the rest of the prepare flow operates on absolute paths as before.

    Three cases:
    - Relative path (post-fix layout): resolve against output_dir
    - Absolute and exists: use as-is
    - Absolute and missing: try basename under output_dir/clips/
      (legacy migration -- handles moved-folder scenario where the saved
      absolute path points at the OLD location but the clip is in the
      new clips/ next to dataset.json)

    Mutates entries in-place. Idempotent.
    """
    clips_dir = output_dir / "clips"
    for e in entries:
        mp = e.get("media_path", "")
        if not mp:
            continue
        p = Path(mp)
        if p.is_absolute():
            if p.exists():
                continue
            candidate = clips_dir / p.name
            if candidate.exists():
                e["media_path"] = str(candidate)
        else:
            e["media_path"] = str((output_dir / p).resolve())


def entries_for_write(entries, output_dir):
    """Return a copy of entries with each media_path stored relative to
    output_dir when the clip lives under it (app-generated artifact).
    Paths outside output_dir stay absolute (user-supplied source data).

    Use forward slashes in the relative form so paths are portable
    between Windows and POSIX. Does NOT mutate the input list -- in
    memory, callers continue to see absolute paths.
    """
    output_dir_resolved = output_dir.resolve()
    out = []
    for e in entries:
        e2 = dict(e)
        mp = e2.get("media_path", "")
        if mp:
            p = Path(mp)
            if p.is_absolute():
                try:
                    rel = p.resolve().relative_to(output_dir_resolved)
                    e2["media_path"] = rel.as_posix()
                except ValueError:
                    pass  # outside output_dir -- keep absolute
        out.append(e2)
    return out


def condition_path_for_clip(output_dir: Path, media_path: Path) -> Path:
    """Derive the condition (.pt) file path for a given clip media_path.

    Mirrors the encoder's path-derivation logic so callers (audit,
    encoder, reject) all look at the same file. When the clip lives
    under output_dir, the path is `<output_dir>/conditions/<rel>.pt`
    preserving the relative subdirectory layout (e.g.
    `conditions/clips/foo.pt`). When it lives outside output_dir,
    falls back to a flat `<output_dir>/conditions/<stem>.pt`.

    Robust to relative media_paths: if media_path isn't absolute, it's
    treated as relative to output_dir before computing the rel path.
    Without this, callers that loaded dataset.json without normalizing
    paths would compute the WRONG output location (flat root, not
    clips/ subdir) and the encoder would write garbage to the root
    conditions folder instead of finding/refreshing the correct file.
    """
    conditions_dir = output_dir / "conditions"
    if not media_path.is_absolute():
        media_path = (output_dir / media_path)
    try:
        rel = media_path.relative_to(output_dir).with_suffix(".pt")
    except ValueError:
        rel = Path(media_path.stem + ".pt")
    return conditions_dir / rel


def latent_path_for_clip(output_dir: Path, media_path: Path) -> Path:
    """Derive the latent (.pt) file path for a given clip media_path.

    Mirrors the encoder's path-derivation logic so callers (audit,
    encoder, reject) all look at the same file. When the clip lives
    under output_dir, the path is `<output_dir>/latents/<rel>.pt`
    preserving the relative subdirectory layout (e.g.
    `latents/clips/foo.pt`). When it lives outside output_dir,
    falls back to a flat `<output_dir>/latents/<stem>.pt`.

    Robust to relative media_paths: if media_path isn't absolute, it's
    treated as relative to output_dir before computing the rel path.
    Without this, callers that loaded dataset.json without normalizing
    paths would compute the WRONG output location (flat root, not
    clips/ subdir) and the encoder would write garbage to the root
    latents folder instead of finding/refreshing the correct file.
    """
    latents_dir = output_dir / "latents"
    if not media_path.is_absolute():
        media_path = (output_dir / media_path)
    try:
        rel = media_path.relative_to(output_dir).with_suffix(".pt")
    except ValueError:
        rel = Path(media_path.stem + ".pt")
    return latents_dir / rel


def audio_latent_path_for_clip(output_dir: Path, media_path: Path) -> Path:
    """Derive the audio_latent (.pt) file path for a given clip media_path.

    Mirrors the encoder's path-derivation logic so callers (audit,
    encoder, reject) all look at the same file. When the clip lives
    under output_dir, the path is `<output_dir>/audio_latents/<rel>.pt`
    preserving the relative subdirectory layout (e.g.
    `audio_latents/clips/foo.pt`). When it lives outside output_dir,
    falls back to a flat `<output_dir>/audio_latents/<stem>.pt`.

    Robust to relative media_paths: if media_path isn't absolute, it's
    treated as relative to output_dir before computing the rel path.
    Without this, callers that loaded dataset.json without normalizing
    paths would compute the WRONG output location (flat root, not
    clips/ subdir) and the encoder would write garbage to the root
    audio_latents folder instead of finding/refreshing the correct file.
    """
    audio_latents_dir = output_dir / "audio_latents"
    if not media_path.is_absolute():
        media_path = (output_dir / media_path)
    try:
        rel = media_path.relative_to(output_dir).with_suffix(".pt")
    except ValueError:
        rel = Path(media_path.stem + ".pt")
    return audio_latents_dir / rel


def purge_clip_artifacts(output_dir: Path, vf: Path) -> None:
    """Delete a clip and all of its precomputed sibling files.

    Removes the clip itself plus its latent / condition / audio_latent
    .pt files under output_dir. Used when a clip is fully rejected
    (e.g., persistent caption-overrun) and must not be re-introduced.

    Idempotent: missing files are silently skipped, deletion failures
    are logged but don't abort.
    """
    for subdir in ("latents", "conditions", "audio_latents"):
        f = output_dir / subdir / "clips" / f"{vf.stem}.pt"
        if f.exists():
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"    Could not delete {f}: {e}")
    if vf.exists():
        try:
            vf.unlink()
        except OSError as e:
            logger.warning(f"    Could not delete clip {vf}: {e}")


def append_rejected(rejected_path: Path, record: dict) -> None:
    """Append a rejection record to <output>/rejected.json.

    Loads the existing list (tolerant of malformed file), appends, and
    writes back. The same load-append-write dance is duplicated in
    many older sites in this file; new code should use this helper.
    """
    rejected: list = []
    if rejected_path.exists():
        try:
            with open(rejected_path) as rf:
                rejected = json.load(rf)
        except (json.JSONDecodeError, KeyError, OSError):
            rejected = []
    rejected.append(record)
    with open(rejected_path, "w") as rf:
        json.dump(rejected, rf, indent=2)


def reject_entry(
    i: int,
    entries: list,
    dataset_json_path: Path,
    reason: str,
    purge_artifacts: bool = False,
) -> dict:
    """Reject the entry at index `i`: pop it, log to rejected.json,
    write dataset.json, and optionally purge on-disk artifacts.

    Returns the popped entry dict so callers can inspect / log fields.

    Args:
        i: Index of the entry to remove from `entries`.
        entries: The dataset entries list (mutated in place).
        dataset_json_path: Path to dataset.json — also used to derive
            rejected.json and output_dir.
        reason: Short string written to rejected.json (e.g.,
            "llm_mismatch", "caption_too_long").
        purge_artifacts: If True, delete the clip + latents + condition
            + audio_latent files for this entry. Use for persistent
            content failures where the clip should not be reused.
    """
    rejected_entry = entries.pop(i)
    output_dir = dataset_json_path.parent
    rejected_path = output_dir / "rejected.json"
    vf = Path(rejected_entry.get("media_path", ""))
    append_rejected(rejected_path, {
        "media_path": str(vf),
        "source_file": rejected_entry.get("source_file", ""),
        "reason": reason,
    })
    if purge_artifacts and vf.name:
        purge_clip_artifacts(output_dir, vf)
    with open(dataset_json_path, "w") as f:
        json.dump(entries, f, indent=2)
    return rejected_entry


def audit_and_repair_dataset(
    output_dir: Path,
    dataset_json_path: Path,
    cond_token_limit: int = COND_TOKEN_LIMIT,
) -> tuple[int, int, int]:
    """Reconcile dataset.json with what's actually on disk.

    Runs once at the top of prepare() to handle state corruption that
    accumulated outside this node's control:
      - Entries whose media_path is missing on disk (true orphans).
      - Clips on disk with no JSON entry (lost-JSON case).
      - Entries whose saved condition file has prompt_attention_mask
        length > cond_token_limit (runaway captions). Blanks the caption
        and deletes the bad condition file so the captioner re-rolls and
        the encoder re-encodes on the next pass.

    Hard rules:
      - Always backs up dataset.json before any write that mutates the
        entry list (timestamped .bak.YYYYMMDD_HHMMSS, matching existing
        convention).
      - Drops entries only when media_path is missing on disk (genuine
        orphan). Never deletes an entry to "fix" a different problem.

    Returns (orphans_dropped, stubs_added, captions_blanked). Caller
    should treat any nonzero count as "audit changed things, do not
    early-exit".
    """
    if not dataset_json_path.exists():
        return (0, 0, 0)

    clips_dir = output_dir / "clips"
    conditions_dir = output_dir / "conditions"

    try:
        with open(dataset_json_path) as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Audit: dataset.json unreadable ({e}); skipping audit")
        return (0, 0, 0)

    if not isinstance(entries, list):
        logger.warning("Audit: dataset.json is not a list; skipping audit")
        return (0, 0, 0)

    normalize_loaded_entries(entries, output_dir)

    orphans: list[str] = []   # entries dropped because media_path missing
    blanked: list[str] = []   # entries whose caption was blanked
    stubs_added = 0

    # Pass 1: drop true orphans (media_path no longer exists)
    surviving = []
    for entry in entries:
        mp = entry.get("media_path", "")
        if not mp:
            # Entry with no media_path is meaningless; treat as orphan
            orphans.append("(no media_path)")
            continue
        if not Path(mp).exists():
            orphans.append(Path(mp).name)
            continue
        surviving.append(entry)
    entries = surviving

    # Pass 2: stub-add clips on disk with no JSON entry
    if clips_dir.exists():
        known = {Path(e["media_path"]).resolve() for e in entries if e.get("media_path")}
        valid_suffixes = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
        for clip_path in sorted(clips_dir.iterdir()):
            if not clip_path.is_file():
                continue
            if clip_path.suffix.lower() not in valid_suffixes:
                continue
            if clip_path.resolve() in known:
                continue
            entries.append({
                "media_path": str(clip_path),
                "source_file": "",
                "characters": [],
            })
            stubs_added += 1

    # Pass 3: detect bad conditions (token length over limit) and blank
    # the corresponding caption + delete the bad condition file. The
    # captioner and encoder are both idempotent: blank caption triggers
    # a re-roll on the next pass, missing condition triggers re-encode.
    if conditions_dir.exists():
        for entry in entries:
            if not entry.get("caption"):
                continue
            mp = Path(entry["media_path"])
            cond_file = condition_path_for_clip(output_dir, mp)
            if not cond_file.exists():
                continue
            try:
                blob = torch.load(cond_file, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(
                    f"Audit: could not load {cond_file.name} ({e}); skipping length check"
                )
                continue
            mask = blob.get("prompt_attention_mask") if isinstance(blob, dict) else None
            if not torch.is_tensor(mask) or mask.dim() < 1:
                continue
            seq_len = int(mask.shape[0])
            if seq_len <= cond_token_limit:
                continue
            # Runaway caption — blank it, delete the bad condition.
            entry["caption"] = ""
            try:
                cond_file.unlink()
            except OSError as e:
                logger.warning(
                    f"Audit: could not delete bad condition file {cond_file.name}: {e}"
                )
            blanked.append(f"{mp.name} ({seq_len} tokens)")

    if not orphans and not blanked and stubs_added == 0:
        return (0, 0, 0)

    # Backup before write. Aborts the audit's writes (but not its
    # in-memory file deletions, which already happened) if the backup
    # fails — better to have a stale dataset.json than a corrupted one.
    timestamp = _t.strftime("%Y%m%d_%H%M%S")
    backup_path = dataset_json_path.with_name(f"dataset.json.bak.{timestamp}")
    try:
        shutil.copy2(dataset_json_path, backup_path)
        logger.info(f"Audit: backed up dataset.json to {backup_path.name}")
    except OSError as e:
        logger.error(
            f"Audit: failed to back up dataset.json ({e}); "
            "skipping audit writes (in-memory entry list will be discarded)"
        )
        return (0, 0, 0)

    with open(dataset_json_path, "w") as f:
        json.dump(entries_for_write(entries, output_dir), f, indent=2)

    if orphans:
        logger.info(
            f"Audit: dropped {len(orphans)} orphan entries (media_path missing on disk)"
        )
        for name in orphans[:10]:
            logger.info(f"  - {name}")
        if len(orphans) > 10:
            logger.info(f"  ...and {len(orphans) - 10} more")
    if stubs_added:
        logger.info(
            f"Audit: added {stubs_added} stub entries for clips on disk with no JSON entry"
        )
    if blanked:
        logger.info(
            f"Audit: blanked {len(blanked)} captions whose condition exceeded "
            f"{cond_token_limit} tokens (will be re-captioned + re-encoded)"
        )
        for name in blanked[:10]:
            logger.info(f"  - {name}")
        if len(blanked) > 10:
            logger.info(f"  ...and {len(blanked) - 10} more")

    return (len(orphans), stubs_added, len(blanked))
