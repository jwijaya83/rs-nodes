# Dataset Self-Heal & Quota Recovery — Spec

Status: planned, not yet implemented.

## Problem

Three failure modes the prepare-dataset pipeline currently leaves the user to handle by hand:

1. **Runaway captioner output**. Probabilistic VLMs (Gemma4 etc.) occasionally
   emit page-long captions for a single clip. The text encoder pads to
   multiples of 128 tokens, so a single rogue caption produces a (2048, 4096)
   condition tensor instead of (128, 4096) — 16× the rest. During training
   this blows up attention memory ~256× on that step, causing intermittent
   C-level aborts whenever the bad sample is randomly drawn. Concrete
   incident: PeeWeePlayhouse `S2 Ep 08 Spring chunk0088` shipped with a
   2048-token caption while every other sample sat at 128 tokens.

2. **Partial dataset corruption**. `dataset.json` or individual condition /
   latent files can be lost or corrupted by other workflow steps. Today the
   user has to restart from scratch, re-captioning every clip even though
   most are intact.

3. **Captioning rejection leaving quota under**. Clips can be rejected during
   captioning (face MISMATCH QC, soon also length-rejection). The mining
   loop runs once before captioning and never re-checks quota, so rejections
   leave the dataset permanently under quota even when the chunk pool has
   unused candidates. With `max_samples` set this matters — quota should be
   met or the pool should be exhausted, whichever comes first.

## Behavior

### A. Audit + repair phase (start of `prepare()`)

Runs once at the top of `prepare()`, before any other work.

1. **Backup `dataset.json`** to `dataset.json.bak.YYYYMMDD_HHMMSS`. Never
   modify entries without a fresh backup first.
2. **Reconcile entries vs disk**:
   - Entry's `media_path` no longer exists on disk → drop entry (true
     orphan).
   - Clip on disk with no matching entry → add stub entry (`media_path` from
     filename, `source_file` derived from name).
3. **Per surviving entry, validate pieces**:
   - Caption present + condition file exists + condition's
     `prompt_attention_mask.shape[0] > 128` → blank caption (set to `""`),
     delete the bad condition file. Captioner will re-roll, encoder will
     re-encode.
   - Transcript missing + `transcribe_speech=True` → flagged for the existing
     transcript-backfill flow (no new code needed, just make sure these
     entries flow through).
   - Missing condition / latent / audio_latent files → handled by the
     existing idempotent encoders; no special action.
4. Hand off to normal mine → caption → encode flow.

### B. Captioner length-check + re-roll

In `_caption_dataset_json`, around the `_caption_single_ollama` call:

1. After each caption is produced, count `len(caption)`.
2. If `len(caption) > MAX_CAPTION_CHARS` (default 600 ≈ 128 tokens after BPE):
   - Wipe Ollama conversation context (`ollama_messages = None`).
   - Log a warning with the first 200 chars of the bad caption.
   - Re-roll.
3. Up to `MAX_CAPTION_RETRIES = 3` attempts.
4. After 3 failures: reject the clip cleanly via the existing `rejected.json`
   + entry-removal path used for MISMATCH. Delete its `clips/`, `latents/`,
   `audio_latents/` files so it doesn't haunt the dataset.

### C. Post-captioning quota warning (deferred: full auto-remine)

**Implemented**: post-captioning quota check that logs a clear warning
when the dataset is under `max_samples` and the persisted `chunk_pool.json`
still has unused candidates. The user re-runs `prepare_dataset` manually;
the audit phase + idempotent encoders pick up exactly where things left
off, the mining loop pops more from the pool until quota is met or the
pool is exhausted.

**Deferred**: fully automatic mine→caption→mine cycling within a single
`prepare()` invocation. The mining flow is interleaved with character-mode
setup, multi-character quota balancing, transcript backfill, and dedup —
all currently inline in `prepare()`. Cleanly extracting it to make it
re-callable would require lifting several hundred lines into helper
methods, with risk of regressions in well-tested existing paths. The
manual re-run path is cheap (one click, audit picks up the state) and the
scenario where this matters in practice (length-rejection AND chunk_pool
non-empty AND quota under) is rare given B's 3-attempt re-roll handles
nearly all runaway captions.

If we revisit auto-remine, the design is:

```text
pass = 0
while pass < MAX_OUTER_PASSES:                # default 3
    pass += 1
    mine clips → quota target                 # existing inner loop, extracted to method
    if pass == 1:
        run post-mining cleanup               # transcripts, dedup, char balance
    free mining-phase models, load Ollama
    caption new entries                       # with B's length-check active
    free Ollama
    if not chunk_pool or not _any_under_quota():
        break
    if pass < MAX_OUTER_PASSES:
        reload mining-phase models
log warning if exit while still under quota
```

VRAM dance: each outer pass beyond the first pays ~30s to free Ollama,
reload face/clipvision/etc., free those, reload Ollama. Cap of 3 outer
passes bounds this at ~2 cycles in the worst case.

## Hard rules

- **Always backup `dataset.json`** before any write that modifies the entry
  list. Timestamped `.bak.YYYYMMDD_HHMMSS` (matches existing convention).
- **Deletion of entries is allowed only when the underlying media file is
  missing on disk** (genuine orphan). Other failure modes use blank-caption
  semantics or full clip rejection (which moves all four files to rejected
  state, not just the entry).
- **Length check is in tokens via mask shape** in the audit phase
  (`prompt_attention_mask.shape[0]`) — that's authoritative. Char-count is a
  fast pre-filter during captioning before the encode step.
- **Idempotency is preserved**: each phase only does work when the
  corresponding output is missing/invalid. Re-running `prepare_dataset`
  never duplicates effort.

## Constants (top of node, easy to tune)

```python
MAX_CAPTION_CHARS = 600       # ~128 tokens after BPE; under the 1921+ that broke training
MAX_CAPTION_RETRIES = 3       # re-roll attempts before clip rejection
MAX_OUTER_PASSES = 3          # mine→caption cycles before giving up on quota
COND_TOKEN_LIMIT = 128        # encoder pad multiple, also audit length-rejection threshold
```

## Out of scope

- Detecting bad captions that are within length limits but semantically
  wrong (hallucinations). The existing MISMATCH face-QC handles the most
  common case; full caption-quality validation needs a separate pass.
- Re-running the entire pipeline if the user changes captioning model or
  settings mid-stream — they're expected to nuke `conditions/` manually if
  they want full re-encoding.
- Cross-character quota balancing during re-mine. The re-mine simply pulls
  from `chunk_pool` and accepts whatever passes the existing quota-aware
  intake gate. Per-character rebalancing happens in the existing dedup pass
  on the first mining loop only.
