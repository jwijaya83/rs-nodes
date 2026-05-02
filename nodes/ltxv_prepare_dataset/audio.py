"""Audio / speech processing for the prepare-dataset node.

Provides whisper transcription, demucs vocal isolation, speechbrain
ECAPA-TDNN voice embedding, and per-speaker attribution. All state
(loaded models) lives in this module's globals; call `unload_audio_models()`
to free GPU memory before captioning or training.
"""

import gc
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Whisper speech transcription (lazy-loaded)
_whisper_model = None
_whisper_model_size_loaded: str | None = None
# Default to large-v3 (~3GB) for best transcription quality.  Smaller sizes
# (base, small, medium) hallucinate badly on music/SFX/chants which is exactly
# the failure mode that pollutes character voice-attribution downstream.
# The user can override via the prep node's `whisper_model` input.
_WHISPER_MODEL_SIZE = "large-v3"

# Demucs vocal isolation (lazy-loaded)
_demucs_model = None
_demucs_device = None

# Speechbrain ECAPA-TDNN speaker embedding (lazy-loaded).
# Used for voice-attribution: embed each Whisper segment and match to enrolled
# voice references via cosine distance.  No HF token required, no torchcodec.
_speechbrain_embedder = None
_VOICE_MATCH_THRESHOLD = 0.70  # cosine distance — lower = stricter
_VOICE_MATCH_MARGIN = 0.05     # require winner to beat runner-up by this much
_FACE_HINT_BONUS = 0.15        # subtract from distance when character is on-screen
                                # (per face detection) — biases ambiguous matches
                                # toward the visible character without overriding
                                # strong off-screen voice evidence.


def get_whisper_model(size: str | None = None):
    """Load Whisper model on first use. Cached after first call.

    If size differs from the cached model's size, the old model is dropped
    and the requested size is loaded fresh.
    """
    global _whisper_model, _whisper_model_size_loaded
    target = size or _WHISPER_MODEL_SIZE
    if _whisper_model is not None and _whisper_model_size_loaded == target:
        return _whisper_model
    if _whisper_model is not None and _whisper_model_size_loaded != target:
        # Different size requested — drop the old model and reload.
        logger.info(f"Whisper model size changed: {_whisper_model_size_loaded} -> {target}, reloading")
        del _whisper_model
        _whisper_model = None
    try:
        import whisper
        logger.info(f"Loading Whisper model ({target})...")
        _whisper_model = whisper.load_model(target)
        _whisper_model_size_loaded = target
        logger.info(f"Whisper model loaded ({target})")
        return _whisper_model
    except ImportError:
        logger.error("openai-whisper not installed. Run: pip install openai-whisper")
        return None


def isolate_vocals(clip_path: Path) -> Path | None:
    """Use Demucs to isolate vocals from a clip's audio.
    Returns path to a temporary WAV file with vocals only,
    or None if demucs isn't available or isolation fails."""
    global _demucs_model, _demucs_device
    try:
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError:
        return None

    try:
        # Load demucs model (cached after first call)
        if _demucs_model is None:
            logger.info("Loading Demucs model for vocal isolation...")
            _demucs_model = get_model("htdemucs")
            _demucs_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Demucs model loaded")

        # Extract audio from clip to WAV
        audio_tmp = clip_path.with_suffix(".tmp_audio.wav")
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(clip_path),
             "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
             str(audio_tmp)],
            capture_output=True, text=True,
        )
        if ffmpeg_result.returncode != 0 or not audio_tmp.exists():
            return None

        # Load and separate
        waveform, sr = torchaudio.load(str(audio_tmp))
        # Demucs expects [batch, channels, samples]
        waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(_demucs_model, waveform, device=_demucs_device)

        # sources shape: [batch, n_sources, channels, samples]
        # htdemucs sources: drums, bass, other, vocals (index 3)
        vocals = sources[0, 3]  # [channels, samples]

        # Save vocals to temp file
        vocals_path = clip_path.with_suffix(".tmp_vocals.wav")
        torchaudio.save(str(vocals_path), vocals.cpu(), sr)

        # Clean up intermediate audio
        try:
            audio_tmp.unlink()
        except OSError:
            pass

        return vocals_path

    except Exception as e:
        logger.warning(f"Demucs vocal isolation failed for {clip_path.name}: {e}")
        # Clean up on failure
        for tmp in [clip_path.with_suffix(".tmp_audio.wav"),
                    clip_path.with_suffix(".tmp_vocals.wav")]:
            try:
                tmp.unlink()
            except OSError:
                pass
        return None


_ECAPA_TARGET_SR = 16000  # speechbrain ECAPA-TDNN expects 16 kHz mono


def get_speechbrain_embedder():
    """Lazy-load speechbrain ECAPA-TDNN speaker-embedding model.
    No HF token required; weights are downloaded and cached on first use."""
    global _speechbrain_embedder
    if _speechbrain_embedder is not None:
        return _speechbrain_embedder
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        logger.error(
            "speechbrain not installed. Run 'pip install speechbrain' in "
            "ComfyUI's venv (or re-run install.bat). Voice attribution disabled."
        )
        return None
    try:
        logger.info("Loading speechbrain ECAPA-TDNN speaker embedder...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Cache weights under the custom-node folder so they're co-located with
        # the rest of the project (and easy to wipe on reinstall).
        savedir = Path(__file__).parent.parent / "models" / "spkrec-ecapa-voxceleb"
        savedir.mkdir(parents=True, exist_ok=True)
        # LocalStrategy.COPY avoids the default symlink behaviour, which fails
        # on Windows without Developer Mode (WinError 1314: A required
        # privilege is not held by the client).  Costs ~80MB extra disk for
        # the duplicate vs the HF cache; portable across all platforms.
        kwargs = dict(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": device},
        )
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass  # older speechbrain — fall through to default symlink
        embedder = EncoderClassifier.from_hparams(**kwargs)
        _speechbrain_embedder = embedder
        logger.info("Speechbrain ECAPA-TDNN loaded")
        return embedder
    except Exception as e:
        logger.error(f"Failed to load speechbrain embedder: {e}")
        return None


def load_audio_for_embedding(audio_path: Path):
    """Load audio as a 1-channel 16 kHz tensor suitable for ECAPA-TDNN.
    Returns (waveform[1, samples], sample_rate=16000) or (None, 0) on failure."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception as e:
        logger.warning(f"Could not load audio {audio_path.name}: {e}")
        return None, 0
    if waveform.numel() == 0:
        return None, 0
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != _ECAPA_TARGET_SR:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, _ECAPA_TARGET_SR)
    return waveform, _ECAPA_TARGET_SR


def embed_audio_segment(audio_path: Path, start: float, end: float) -> np.ndarray | None:
    """Embed audio [start, end] seconds with ECAPA-TDNN.
    Returns 192-d numpy array (L2-normalized) or None on failure."""
    embedder = get_speechbrain_embedder()
    if embedder is None:
        return None
    waveform, sr = load_audio_for_embedding(audio_path)
    if waveform is None:
        return None
    s = max(0, int(start * sr))
    e = min(waveform.shape[1], int(end * sr))
    if e - s < int(0.3 * sr):  # need at least 0.3s for a stable embedding
        return None
    slice_wave = waveform[:, s:e]
    try:
        emb = embedder.encode_batch(slice_wave).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.warning(f"Embedding failed for {audio_path.name} [{start:.2f}-{end:.2f}]: {e}")
        return None


def embed_audio_full(audio_path: Path) -> np.ndarray | None:
    """Embed an entire audio file (used for voice reference enrollment)."""
    embedder = get_speechbrain_embedder()
    if embedder is None:
        return None
    waveform, _ = load_audio_for_embedding(audio_path)
    if waveform is None:
        return None
    try:
        emb = embedder.encode_batch(waveform).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.warning(f"Voice reference embedding failed for {audio_path.name}: {e}")
        return None


def format_speaker_display(trigger: str) -> str:
    """pee-wee → Pee-wee, miss-yvonne → Miss Yvonne, cowboy-curtis → Cowboy Curtis."""
    if not trigger or trigger == "unknown":
        return "Someone"
    parts = trigger.replace("_", "-").split("-")
    return " ".join(p[:1].upper() + p[1:] for p in parts if p)


def transcribe_clip(clip_path: Path, voice_refs: dict | None = None,
                     face_chars: set | None = None) -> dict | None:
    """Run Demucs vocal isolation then Whisper on a clip.
    Returns dict with 'text', 'words' list of {word, start, end}, 'duration'.
    Returns None if transcription fails or no speech detected.

    When voice_refs is provided, also runs pyannote speaker diarization +
    embedding to attribute each segment to a known speaker.  In that case
    the returned dict additionally contains 'segments': list of {speaker,
    text, start, end} and 'text' is rewritten with speaker labels of the
    form `<Speaker Display Name>: "<line>"`, joined sequentially.  Speakers
    that don't match any enrolled voice above threshold are tagged as
    'unknown' (rendered as "Someone").
    """
    model = get_whisper_model()
    if model is None:
        return None

    import whisper

    # Isolate vocals to remove music/sfx before transcribing.  We keep the
    # vocals file alive across both Whisper and pyannote so diarization and
    # speaker embedding see the same isolated audio Whisper transcribed.
    vocals_path = isolate_vocals(clip_path)
    audio_source = str(vocals_path) if vocals_path else str(clip_path)

    whisper_segs: list[dict] = []
    text = ""
    try:
        try:
            result = whisper.transcribe(model, audio_source, word_timestamps=True)
        except Exception as e:
            logger.warning(f"Whisper transcription failed for {clip_path.name}: {e}")
            return None

        text = result.get("text", "").strip()
        if not text:
            return None

        # Detect Whisper hallucinations — non-Latin gibberish when there's no speech
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and latin_chars / total_alpha < 0.5:
            logger.info(f"  Whisper hallucination detected, no usable speech: {text[:60]}")
            return {"text": "", "words": [], "duration": 0.0, "hallucination": True}

        # Extract word-level timestamps
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                })

        # Get clip duration via ffprobe
        duration = 0.0
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(clip_path)],
                capture_output=True, text=True,
            )
            duration = float(probe.stdout.strip())
        except (ValueError, OSError):
            pass

        whisper_segs = [
            {"start": float(s["start"]), "end": float(s["end"]),
             "text": s.get("text", "").strip()}
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]

        # No voice references → return flat-text format (legacy callers).
        if not voice_refs:
            return {"text": text, "words": words, "duration": duration}

        # Voice attribution: embed each Whisper segment with ECAPA-TDNN and
        # match against enrolled voice prints.  When face_chars is provided
        # (the visible characters from face detection), it biases ambiguous
        # matches toward on-screen speakers.
        embedded_audio = Path(audio_source)
        segments = attribute_whisper_segments(
            embedded_audio, whisper_segs, voice_refs, face_chars=face_chars,
        )
        if segments is None:
            # Diarization unavailable — fall back to flat text but flag it.
            logger.info(
                f"  Voice attribution unavailable for {clip_path.name} — "
                f"emitting flat transcript without speaker labels"
            )
            return {"text": text, "words": words, "duration": duration}

        # Rebuild flat text with `<Speaker>: "<line>"` labels so any consumer
        # reading the legacy 'text' field still gets the speaker context.
        flat_parts = []
        for s in segments:
            sp_disp = format_speaker_display(s["speaker"])
            line = s["text"].strip().rstrip(".!?")
            flat_parts.append(f'{sp_disp}: "{line}".')
        flat_text = " ".join(flat_parts) if flat_parts else text

        return {
            "text": flat_text,
            "words": words,
            "duration": duration,
            "segments": segments,
        }
    finally:
        # Clean up vocals temp file (after both Whisper AND any diarization)
        if vocals_path:
            try:
                vocals_path.unlink()
            except OSError:
                pass


def attribute_whisper_segments(
    audio_path: Path, whisper_segs: list[dict], voice_refs: dict,
    face_chars: set | None = None,
) -> list[dict] | None:
    """For each Whisper segment, embed its audio slice and match against
    enrolled voice_refs via cosine distance. Returns one entry per Whisper
    segment tagged with the matched speaker name (or 'unknown').

    Whisper's segments are already speech-bounded, so we don't need a
    separate diarization pass — each segment gets independently attributed.
    Trade-off: segments that contain TWO speakers (overlap) get a single
    label.  In practice, Whisper's segmenter cuts on speaker change boundaries
    fairly often, and our use case has mostly single-speaker clips anyway.

    Returns None if the embedder isn't available — caller falls back to
    flat-text transcripts.
    """
    embedder = get_speechbrain_embedder()
    if embedder is None:
        return None

    waveform, sr = load_audio_for_embedding(audio_path)
    if waveform is None:
        return None

    segments_out: list[dict] = []
    for ws in whisper_segs:
        s_frame = max(0, int(ws["start"] * sr))
        e_frame = min(waveform.shape[1], int(ws["end"] * sr))
        slice_dur = (e_frame - s_frame) / sr if sr else 0.0
        speaker = "unknown"
        match_diag = ""
        if slice_dur >= 0.3:  # need ~0.3s for a stable embedding
            try:
                slice_wave = waveform[:, s_frame:e_frame]
                emb = embedder.encode_batch(slice_wave).squeeze().detach().cpu().numpy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                    # Compute raw + face-hint-adjusted distance per candidate.
                    # Face hint subtracts a small bonus when the character is
                    # on-screen for this clip, biasing ambiguous matches toward
                    # visible speakers without overriding strong off-screen
                    # voice evidence.
                    scored = []
                    for name, ref_emb in voice_refs.items():
                        raw = 1.0 - float(np.dot(emb, ref_emb))
                        adj = raw - _FACE_HINT_BONUS if (face_chars and name in face_chars) else raw
                        scored.append((name, raw, adj))
                    # Rank by adjusted distance (lower = better match)
                    scored.sort(key=lambda p: p[2])
                    best_name, best_raw, best_adj = scored[0]
                    runner_adj = scored[1][2] if len(scored) > 1 else float("inf")
                    margin = runner_adj - best_adj
                    if best_adj <= _VOICE_MATCH_THRESHOLD and margin >= _VOICE_MATCH_MARGIN:
                        speaker = best_name
                    # Log raw and adjusted (face-hinted) distances so threshold
                    # tuning is informed by real data.
                    summary = ", ".join(
                        f"{n}={r:.2f}" + (f"→{a:.2f}*" if a != r else "")
                        for n, r, a in scored
                    )
                    match_diag = (
                        f" [match: best={best_name}@{best_adj:.2f} "
                        f"margin={margin:.2f} → {speaker}; all: {summary}]"
                    )
            except Exception as e:
                logger.warning(f"Speaker match failed for segment "
                               f"[{ws['start']:.2f}-{ws['end']:.2f}]: {e}")
        if match_diag:
            logger.info(f"  seg [{ws['start']:.1f}-{ws['end']:.1f}]{match_diag}")
        segments_out.append({
            "speaker": speaker,
            "text": ws["text"],
            "start": ws["start"],
            "end": ws["end"],
        })
    return segments_out


def unload_audio_models():
    """Free all audio-side models this module holds: Whisper, Demucs, and
    speechbrain ECAPA-TDNN. Safe to call even if nothing was loaded — each
    unload is guarded.

    Returns the list of names freed, so the caller can include them in a
    final unload log.
    """
    global _whisper_model, _demucs_model, _demucs_device
    global _speechbrain_embedder

    freed = []
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        freed.append("Whisper")
    if _demucs_model is not None:
        del _demucs_model
        _demucs_model = None
        _demucs_device = None
        freed.append("Demucs")
    if _speechbrain_embedder is not None:
        try:
            del _speechbrain_embedder
        except Exception:
            pass
        _speechbrain_embedder = None
        freed.append("speechbrain-ecapa")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return freed
