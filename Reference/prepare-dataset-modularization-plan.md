# `ltxv_prepare_dataset` Modularization Plan

Status: planned, not yet executed.

## Problem

`nodes/ltxv_prepare_dataset.py` is **5803 lines** in a project where the
next-biggest node file is 2343. It mixes seven distinct concerns into one
class, violating the project's "modules over functions" rule.

## Approach

Convert the single-file module `nodes/ltxv_prepare_dataset.py` into a
**package** at `nodes/ltxv_prepare_dataset/` so the existing import path
`from .nodes.ltxv_prepare_dataset import RSLTXVPrepareDataset` keeps
working without changes anywhere outside the package.

The class stays as the orchestrator. Pure-utility functions and concern-
specific blocks become standalone functions in sibling modules. The class
calls them; it does not own them.

The refactor also inverts the current entry-driven flow into an
**artifact-driven reconciliation flow** (next section). This is the
correct model and avoids a class of bugs where dataset.json fields and
on-disk artifacts disagree.

## Artifact-driven reconciliation flow

### Principle

`dataset.json` is the source of truth **only for encoding**. Once clips
are encoded into latents / conditions / audio_latents, the encoded files
ARE the dataset for training. JSON metadata (captions, transcripts,
character lists) only matters when there's a missing or invalid encoded
file that needs to be (re-)produced from it.

If all encoded artifacts are present and consistent on disk, JSON state
is irrelevant. The system must not re-process anything based purely on
"this entry has no transcript field" or "this entry has no caption" if
the encoded condition / audio_latent is already there.

### Flow

```text
prepare() entry
├── 1. reconcile_artifacts(output_dir, with_audio) -> ReconciliationReport
│       (cheap, disk-only, no JSON read)
│       returns:
│         needed_stems         # set of clip stems on disk
│         missing_latents      # stems with clip but no latent
│         missing_conditions   # stems with clip but no condition
│         missing_audio_latents# stems with clip but no audio_latent (if with_audio)
│         orphan_artifacts     # latents/conditions/audio_latents without a matching clip
│         bad_conditions       # condition files with mask shape > COND_TOKEN_LIMIT
│
├── 2. scan_media(media_folder) -> new source clips not yet in clips/
│
├── 3. if no missing artifacts AND no orphans AND no bad conditions
│        AND no new source media:
│          return early — training is unblocked, JSON not read at all.
│
├── 4. otherwise enter targeted-repair mode:
│      a. read dataset.json (needed for captions / transcripts of missing
│         conditions; needed for media_path lookup of orphan clips)
│      b. audit dataset.json against reconciliation report:
│         - drop entries whose clip is gone (orphan in JSON)
│         - add stub entries for clip stems present on disk but absent
│           from JSON (lost-JSON case)
│         - blank captions for entries pointing at bad_conditions
│           (audit deletes the bad cond file, captioner+encoder will
│           refill on next pass)
│         - delete orphan_artifacts (latent/condition/audio_latent files
│           with no clip — wasted disk)
│      c. for new_source_media: run mining on those source files only
│      d. for missing_conditions: run captioning on those stems only
│         (entries whose caption is empty get captioned; entries whose
│         caption is present skip straight to encoding)
│      e. for missing_latents: run latent encoding on those stems only
│      f. for missing_audio_latents: run audio_latent encoding on those
│         stems only
│      g. for missing_conditions: run text encoding on those stems only
│
└── 5. final cleanup, return.
```

Phases d–g become **input-list-driven** rather than scan-everything: each
phase is told exactly which stems to process, not "scan dataset.json for
anything that needs it". That eliminates the class of bug where a JSON
entry is missing a field but the corresponding artifact is fine — the
artifact wins, no work is done.

### What changes vs. today

- `prepare()` no longer iterates entries to decide what work to do; it
  iterates *missing artifacts* and looks up entries only when needed.
- The transcript backfill phase becomes part of mining (transcribe runs
  during clip extraction, when the clip is being created, and the result
  is written into the new entry). It is not re-run on existing entries
  with empty transcript fields — that's exactly the entry-vs-artifact
  mismatch we're trying to escape.
- Caption phase only runs for stems in `missing_conditions` whose JSON
  entry has no caption. If a stem is in `missing_conditions` but has a
  caption already, only the encoder runs.
- Orphan artifact files (latent/condition/audio_latent without a clip)
  get cleaned up so they stop confusing future runs.

### Backward compat

Existing datasets work without any user action: the reconciliation
report on a fully-built dataset returns all-empty sets, the early-exit
fires, JSON is never read. Datasets with the JSON-vs-artifact mismatches
that today would silently re-transcribe everything will now correctly
skip those phases.

The only behavior change a user would notice is: re-running
prepare_dataset on a complete dataset is now a no-op even if you've
edited captions in dataset.json. That's intentional — to apply caption
edits, delete the corresponding `conditions/clips/<stem>.pt` and re-run;
the missing condition is detected, the (edited) caption is read from
JSON, the new condition is encoded.

## Target layout

```
nodes/ltxv_prepare_dataset/
    __init__.py          # RSLTXVPrepareDataset class + tuning constants + prepare() orchestrator
    audio.py             # transcription, voice attribution, dialogue alignment
    face.py              # InsightFace + DNN detection, embedding, matching, crops
    captioning.py        # Ollama + Gemma captioning, length-check, prep, normalize
    encoding.py          # text + latent + audio_latent encoding (in-process and subprocess)
    mining.py            # _process_video, _process_image, chunk pool, content rejection
    dataset_io.py        # entries normalize, audit, reject helpers, rejected.json
    status.py            # _emit_prepper_status + status payload helpers
```

## Module breakdown

Source line ranges below are best-effort — the executing agent must verify
them by `grep`-ing for the function definitions, not by trusting the
numbers, since edits during this conversation may have shifted them.

### `__init__.py` (target ~600–800 lines)

Stays:
- `RSLTXVPrepareDataset` class shell
- `INPUT_TYPES`, `IS_CHANGED`, `VALIDATE_INPUTS`
- Module-level tuning constants: `COND_TOKEN_LIMIT`, `MAX_CAPTION_CHARS`,
  `MAX_CAPTION_RETRIES`, `MAX_OUTER_PASSES`
- `prepare()` method — the main orchestrator. Internally rewires every
  call site that previously hit a class method to instead call a function
  imported from a sibling module.
- Imports from each sibling module to expose the call surface used by
  `prepare()`.

Removed (moved to siblings, not deleted from history — `git mv` style):
- Everything else.

### `audio.py` (~430 lines)

Module-level functions to move (pure module-level today, easy lift):
- `_get_whisper_model`, `_isolate_vocals`
- `_get_speechbrain_embedder`, `_load_audio_for_embedding`
- `_embed_audio_segment`, `_embed_audio_full`
- `_format_speaker_display`
- `_transcribe_clip`, `_attribute_whisper_segments`

Module globals to relocate (encapsulate as module-state in `audio.py`):
- `_whisper_model`, `_demucs_model`, `_demucs_device`
- `_speechbrain_embedder`
- `_WHISPER_MODEL_SIZE`

Public surface:
- `transcribe_clip(clip_path, voice_refs=None, ...)`
- `attribute_whisper_segments(...)`
- `embed_audio_full(audio_path)`
- `format_speaker_display(trigger)`
- `unload_audio_models()` — wraps the audio half of
  `_unload_all_prepper_models`

### `face.py` (~400 lines, includes crops)

Module-level functions to move:
- `_get_face_app`, `_analyze_frame`
- `_detect_face_dnn`, `_detect_all_faces_dnn`
- `_get_face_embedding`, `_match_face`, `_has_unknown_face`
- `_compute_face_crop`, `_compute_pan_and_scan`

Module globals to relocate:
- `_face_app`, `_face_app_checked`
- `_last_analysis_frame_id`, `_last_analysis_faces`
- `_FACE_PADDING`

Public surface:
- `detect_faces(frame)`, `detect_face_dnn(frame)`
- `embed_face(frame, rect)`, `match_face(emb, target_emb)`
- `compute_face_crop(...)`, `compute_pan_and_scan(...)`
- `set_face_padding(value)` — replaces the `global _FACE_PADDING`
  assignment in `prepare()`
- `unload_face_models()`

### `captioning.py` (~800 lines)

Class methods to move (convert `self` → module-level, take dependencies
as args):
- `_caption_dataset_json` → `caption_dataset_json(...)`
- `_caption_single_ollama` → `caption_single_ollama(...)`
- `_prepare_clip_for_caption` → `prepare_clip_for_caption(...)`
- `_normalize_caption_for_encode` (classmethod) → `normalize_caption_for_encode(...)`
- `_titlecase_name` (staticmethod) → `titlecase_name(name)`

Constants to relocate:
- `MAX_CAPTION_CHARS`, `MAX_CAPTION_RETRIES` move here (still re-exported
  by `__init__.py` for backward-compat reads, but the live definition is
  next to the code that uses them).
- `_CAPTION_QUOTE_STRIP_RE` and any other caption regex.

Public surface:
- `caption_dataset_json(dataset_json_path, target_stems: set[str], ...)` —
  the main captioning loop, but now scoped to `target_stems`. Iterates
  only entries whose stem is in `target_stems` AND whose caption is empty.
  Entries with captions are not re-captioned even if their stem is in the
  target set (the target set is "needs a condition encoded", which a
  pre-existing caption can satisfy without re-captioning).
- `prepare_clip_for_caption(...)` — frame extraction, ID pass

### `encoding.py` (~400 lines)

Class methods to move:
- `_encode_conditions_inprocess` → `encode_conditions_inprocess(...)`
- `_encode_latents_inprocess` → `encode_latents_inprocess(...)`
- `_load_video_frames` → `load_video_frames(...)`
- `_resolve_resolution_buckets` → `resolve_resolution_buckets(...)`

Constants to relocate:
- `COND_TOKEN_LIMIT` lives here (re-exported from `__init__.py`).

Public surface:
- `encode_conditions_inprocess(clip, dataset_json_path, conditions_dir, target_stems: set[str], ...)`
- `encode_latents_inprocess(vae, dataset_json_path, latents_dir, target_stems: set[str], ...)`
- `encode_audio_latents(...)` (extracted from the with_audio branches)

Each takes `target_stems` and processes only entries matching. No
phase-level "scan everything that's missing" logic — the orchestrator
already knows what's missing from the reconciliation report.

### `mining.py` (~1500 lines, the biggest chunk)

Class methods to move:
- `_process_video` → `process_video(...)`
- `_process_image` → `process_image(...)`
- `_scan_media` → `scan_media(folder)`
- `_compute_target_embedding` → `compute_target_embedding(...)`
- `_load_character_refs`, `_load_voice_refs`, `_load_location_refs`
- `_clip_vision_encode`, `_match_characters_in_frame`, `_check_face_match`
- `_filter_dominant_chars` (classmethod)
- `_flush_consumed_chunks`, `_flush_rejected_chunks`
- `_quarantine_clip` (staticmethod), `_record_clip_rejection` (staticmethod)

Public surface:
- `process_video(...)`, `process_image(...)` — single-file workers
- `process_new_media(media_files, output_dir, ...)` — orchestrates mining
  for a list of source files. The current `prepare()` mining inner loop
  (around lines 1849+) moves here in full, including the chunk_pool +
  per-character quota balancing logic. Returns the new entries to append
  to dataset.json.
- `scan_media(folder)`
- `load_character_refs(folder, clip_vision=None)`
- `load_voice_refs(folder)`, `load_location_refs(folder)`
- `compute_target_embedding(target_face_tensor)`

Note: the mining inner loop moves OUT of `__init__.py` (different from
the original plan above). After the reconciliation refactor, mining is
called only when there's new source media; it's no longer woven into the
main flow as a default phase.

### `dataset_io.py` (~400 lines)

Class methods + helpers to move:
- `_normalize_loaded_entries` (staticmethod)
- `_entries_for_write` (staticmethod)
- `_audit_and_repair_dataset` — reduced in scope, becomes a step inside
  the reconciliation flow rather than a standalone gatekeeper
- `_condition_path_for_clip` (staticmethod)
- `_purge_clip_artifacts` (staticmethod)
- `_append_rejected` (staticmethod)
- `_reject_entry`

New functions to add (the reconciliation primitives):
- `reconcile_artifacts(output_dir, with_audio, cond_token_limit) -> ReconciliationReport`
- `apply_audit_to_json(report, output_dir, dataset_json_path) -> AuditResult`
- `delete_orphan_artifacts(report, output_dir)`

Public surface:
- `normalize_loaded_entries(entries, output_dir)`
- `entries_for_write(entries, output_dir)`
- `condition_path_for_clip(output_dir, media_path)`
- `latent_path_for_clip(output_dir, media_path)` (new — same pattern)
- `audio_latent_path_for_clip(output_dir, media_path)` (new — same pattern)
- `purge_clip_artifacts(output_dir, vf)`
- `append_rejected(rejected_path, record)`
- `reject_entry(i, entries, dataset_json_path, reason, purge_artifacts=False)`
- `reconcile_artifacts(output_dir, with_audio, cond_token_limit)` — the
  primary entry point for the new flow
- `apply_audit_to_json(report, output_dir, dataset_json_path)` — uses
  the report to drop orphan entries / stub missing entries / blank
  captions for bad conditions
- `delete_orphan_artifacts(report, output_dir)` — clean up artifacts
  with no matching clip

This is the cleanest module to extract first — almost all of it is pure
functions already, and most of the new self-heal code lives here.

### `ReconciliationReport` shape

```python
@dataclass
class ReconciliationReport:
    needed_stems: set[str]           # all clip stems present in clips/
    missing_latents: set[str]         # stems with clip but no latent
    missing_conditions: set[str]      # stems with clip but no condition
    missing_audio_latents: set[str]   # stems with clip but no audio_latent
                                      # (always empty if with_audio=False)
    bad_conditions: set[str]          # stems whose condition file has
                                      # prompt_attention_mask shape > limit
    orphan_latents: set[str]          # latent stems with no clip
    orphan_conditions: set[str]       # condition stems with no clip
    orphan_audio_latents: set[str]    # audio_latent stems with no clip

    @property
    def fully_consistent(self) -> bool:
        """True iff every needed stem has every artifact AND there are no
        orphan artifacts AND no bad conditions. When True, prepare() can
        early-exit without reading dataset.json at all."""
        return (
            not self.missing_latents
            and not self.missing_conditions
            and not self.missing_audio_latents
            and not self.bad_conditions
            and not self.orphan_latents
            and not self.orphan_conditions
            and not self.orphan_audio_latents
        )
```

### `prepare()` orchestrator skeleton (in `__init__.py`)

```python
def prepare(self, ...):
    # ... arg parsing, model loading ...

    # 1. Disk-only reconciliation. No JSON read.
    report = dataset_io.reconcile_artifacts(
        output_dir, with_audio=with_audio, cond_token_limit=COND_TOKEN_LIMIT,
    )

    # 2. Scan source media for new clips not yet ingested.
    media_files = mining.scan_media(media_folder)
    new_source_media = [m for m in media_files
                         if Path(m["path"]).stem not in
                            {s.rsplit("_chunk", 1)[0] for s in report.needed_stems}]

    # 3. Early-exit: nothing to do.
    if report.fully_consistent and not new_source_media:
        logger.info(f"prepare: dataset complete ({len(report.needed_stems)} stems), "
                    f"skipping all phases")
        return (str(output_dir), str(dataset_json_path))

    # 4. Targeted repair mode.
    audit = dataset_io.apply_audit_to_json(report, output_dir, dataset_json_path)
    dataset_io.delete_orphan_artifacts(report, output_dir)

    if new_source_media:
        mining.process_new_media(new_source_media, ...)  # creates new entries + clips

    # Caption only entries that lack a caption AND whose condition is missing.
    # Stems with caption already + missing condition skip captioning, go straight to encoding.
    caption_targets = report.missing_conditions  # stems needing condition
    captioning.caption_dataset_json(
        dataset_json_path, target_stems=caption_targets, ...
    )

    # Encode only the missing artifacts.
    if report.missing_conditions:
        encoding.encode_conditions_inprocess(
            clip, dataset_json_path, conditions_dir,
            target_stems=report.missing_conditions,
        )
    if report.missing_latents:
        encoding.encode_latents_inprocess(
            vae, dataset_json_path, latents_dir,
            target_stems=report.missing_latents,
        )
    if report.missing_audio_latents:
        encoding.encode_audio_latents(...)

    # 5. Cleanup, unload, return.
    self._unload_all_prepper_models()
    return (str(output_dir), str(dataset_json_path))
```

Each phase function takes `target_stems` as a required argument. No
phase scans dataset.json to find work — the orchestrator hands them
exactly what to process.

### `status.py` (~50 lines)

Move:
- `_emit_prepper_status` (module-level)

Public surface:
- `emit_prepper_status(node_id, char_counts, total_clips, max_samples, ...)`

## Cross-cutting concerns

### `_unload_all_prepper_models`

Currently a single staticmethod that calls into every backend's unload.
After the split, replace it with a single function in `__init__.py` that
calls each module's `unload_*` function:

```python
def unload_all_prepper_models():
    audio.unload_audio_models()
    face.unload_face_models()
    # captioning.py uses Ollama (separate process) — no PyTorch models to unload
    # encoding.py uses passed-in clip/vae — caller manages those
```

### Module-level globals

Each domain's globals (e.g., `_face_app`, `_whisper_model`) currently
live at the top of the file. After the split they live at the top of
their respective module. `_unload_all_prepper_models` must reach into
each module's globals via that module's unload function, not via
`global` keyword from `__init__.py`.

### Imports

`__init__.py` only imports siblings as needed (lazy is fine if any
import is heavy). Sibling modules import from each other minimally:
- `mining` → `face`, `audio` (it uses face detection + transcription)
- `captioning` → none (Ollama interface is self-contained)
- `encoding` → none (uses passed-in models)
- `dataset_io` → `encoding` only for the path helper (or pull the path
  helper UP to a tiny `paths.py` if the cycle gets ugly)

### What `self` arguments become

When converting `def _foo(self, ...)` to module-level `def foo(...)`:
- If `self.<attr>` is read but never written: pass as keyword arg.
- If a method uses `self._clip_characters` or similar instance state:
  pass the dict directly, e.g., `clip_characters: dict`.
- Static / classmethods drop the receiver and become free functions.

## Execution plan (parallel agents)

Each agent gets one module. Agent prompt template:

> Extract the following functions from
> `nodes/ltxv_prepare_dataset.py` into a new module
> `nodes/ltxv_prepare_dataset/<module>.py`. Convert `self.<method>`
> calls to module-level function calls; convert `self.<attr>` reads
> to function arguments. Preserve all logic and comments. Do **not**
> modify `nodes/ltxv_prepare_dataset.py` itself yet — only create
> the new file. Functions to move: [list].
> Public surface: [list]. Module globals to relocate: [list].

Order:
1. **Phase 1 (parallel)**: agents create `audio.py`, `face.py`,
   `captioning.py`, `encoding.py`, `dataset_io.py`, `status.py`. None
   of these touch the original file yet.
2. **Phase 2 (sequential, single agent)**: convert
   `nodes/ltxv_prepare_dataset.py` to
   `nodes/ltxv_prepare_dataset/__init__.py`, importing from siblings
   and replacing every method body with a thin call to the new
   function. `mining.py` extracted last because it's the biggest and
   the orchestrator references it most.
3. **Phase 3 (sequential, single agent)**: import-order debug pass.
   Run `python -c "from custom_nodes.rs_nodes import *"` (or
   equivalent), fix any circular imports or missing names. Run
   `py_compile` on every module.
4. **Phase 4 (sequential, single agent)**: smoke test by loading
   ComfyUI and running a tiny prepare_dataset call, verify it
   completes without import errors. (User-driven, since it requires
   ComfyUI startup.)

## Validation

After the split, the file count change should be:
- `nodes/ltxv_prepare_dataset.py`: deleted (replaced by package).
- `nodes/ltxv_prepare_dataset/`: new package with 7 files totaling
  roughly the same line count as the original ±~5% (helper imports
  + module docstrings).

Behavior must be identical for the happy path. Tests:
- `from custom_nodes.rs_nodes import RSLTXVPrepareDataset` succeeds.
- `RSLTXVPrepareDataset.INPUT_TYPES()` returns the same dict as before.
- A fresh prepare_dataset run on a small dataset produces byte-identical
  output `dataset.json`, `latents/`, `conditions/` to a pre-refactor run.

Reconciliation-flow tests (these are NEW behavior, expected to differ
from pre-refactor):
- Run prepare_dataset on a complete dataset → returns in <1s, no
  transcription, no captioning, no encoding. Pre-refactor would have
  re-run the full pipeline if dataset.json had any field gaps.
- Delete one `conditions/clips/<stem>.pt` → re-run prepare_dataset →
  only that one stem gets re-encoded. Caption is read from existing
  dataset.json entry, not re-captioned.
- Delete one `latents/clips/<stem>.pt` → only that stem gets re-encoded
  for latents. Captioning + condition phases skip entirely.
- Stub case: copy a clip into `clips/` without adding a JSON entry →
  re-run → audit adds stub entry, captioner runs only on that one stem,
  encoders run only on that one stem.
- Orphan case: leave a `latents/clips/<stem>.pt` whose clip is gone →
  re-run → orphan latent is deleted, no other work runs.
- Bad-condition case: replace a `conditions/clips/<stem>.pt` with a file
  whose `prompt_attention_mask.shape[0] > COND_TOKEN_LIMIT` → re-run →
  caption is blanked, captioner re-rolls only that one, encoder re-runs
  only that one.

## Out of scope

- Cleaning up the 6 remaining pre-existing `rejected.json` duplications
  in older sites (separate task; trivial after `dataset_io.append_rejected`
  is in place).
- Refactoring `ltxv_generate.py` (2343 lines, also bloated). Same approach
  would apply, planned as a follow-up.
- Pulling shared LTXV helpers out of `nodes/` and into `utils/` — many of
  the encoding helpers are also useful to `ltxv_generate` and friends.
  Worth doing, but a separate plan.
