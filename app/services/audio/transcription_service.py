from __future__ import annotations

import logging
import uuid

from app.core.config import Settings
from app.services.audio.conversion_service import AudioConversionService
from app.services.audio.lyrics_aligner import align_to_lrc
from app.services.audio.model_manager import get_whisper_model

# Whisper transcription utilities. Uses Groq's hosted Whisper Large v3 when a
# GROQ_API_KEY is configured (faster + frees local RAM), falling back to the
# local openai-whisper model otherwise (or if the Groq call fails).

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, settings: Settings, conversion_service: AudioConversionService) -> None:
        self._settings = settings
        self._conversion = conversion_service

    def transcribe_lrc(
        self,
        audio_bytes: bytes,
        title: str = "",
        artist: str = "",
        *,
        model_name: str | None = None,
        device: str | None = None,
        beam_size: int | None = None,
        best_of: int | None = None,
        temperature: float = 0.0,
        use_word_timestamps: bool = True,
        hint_lyrics: str | None = None,
    ) -> str:
        """Plain transcription -> LRC (words come from the speech-to-text)."""
        context_prompt = self._build_prompt(title, artist, hint_lyrics)
        try:
            segments = self._get_segments(
                audio_bytes,
                context_prompt,
                use_word_timestamps=use_word_timestamps,
                temperature=temperature,
                model_name=model_name,
                device=device,
                beam_size=beam_size,
                best_of=best_of,
            )
            return self._build_lrc(segments, use_word_timestamps)
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return ""

    def transcribe_aligned_lrc(
        self,
        audio_bytes: bytes,
        reference_lyrics: str,
        title: str = "",
        artist: str = "",
        *,
        model_name: str | None = None,
        device: str | None = None,
        beam_size: int | None = None,
        best_of: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Aligns the user's reference lyrics to the transcription timings.

        The WORDS come from ``reference_lyrics`` (100% correct) and the TIMES from
        the transcription. Falls back to plain transcription if alignment is not
        possible. Word timestamps are always requested (needed to align)."""
        context_prompt = self._build_prompt(title, artist, reference_lyrics)
        try:
            segments = self._get_segments(
                audio_bytes,
                context_prompt,
                use_word_timestamps=True,
                temperature=temperature,
                model_name=model_name,
                device=device,
                beam_size=beam_size,
                best_of=best_of,
            )
            words = [
                {"start": w["start"], "word": w["word"]}
                for seg in segments
                for w in (seg.get("words") or [])
            ]
            if words:
                aligned = align_to_lrc(reference_lyrics, words)
                if aligned.strip():
                    logger.info("Lyrics aligned to reference (%d words).", len(words))
                    return aligned
            logger.warning("Could not align lyrics; falling back to plain transcription.")
            return self._build_lrc(segments, True)
        except Exception as exc:
            logger.error("Aligned transcription error: %s", exc)
            return ""

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _build_prompt(self, title: str, artist: str, hint_lyrics: str | None) -> str:
        # Whisper/Groq use the prompt as vocabulary/style context, so we prime it
        # with the song's name and (when available) a sample of the real lyrics.
        parts: list[str] = []
        if title and artist:
            parts.append(f"Cancion: {title} - {artist}.")
        elif title:
            parts.append(f"Cancion: {title}.")
        parts.append("Letra de cancion en espanol, con puntuacion natural.")
        if hint_lyrics:
            snippet = " ".join(hint_lyrics.split())[:600]
            parts.append(f"Letra de referencia: {snippet}")
        return " ".join(parts)

    def _build_lrc(self, segments: list[dict], use_word_timestamps: bool) -> str:
        """Builds the LRC string from a normalized list of segments
        ``[{"start": float, "text": str, "words": [{"start": float, "word": str}]}]``.
        Output format matches what the frontend's parseLrc expects."""
        lrc_lines: list[str] = []
        for segment in segments:
            start = float(segment.get("start") or 0.0)
            m = int(start // 60)
            s = int(start % 60)
            c = int((start % 1) * 100)

            line_text = ""
            words = segment.get("words") or []
            if use_word_timestamps and words:
                for w in words:
                    ws = float(w.get("start") or 0.0)
                    wm = int(ws // 60)
                    ws_s = int(ws % 60)
                    wc = int((ws % 1) * 100)
                    word_clean = (w.get("word") or "").strip()
                    if word_clean:
                        line_text += f"<{wm:02d}:{ws_s:02d}.{wc:02d}> {word_clean} "
            else:
                line_text = (segment.get("text") or "").strip()

            if line_text.strip():
                lrc_lines.append(f"[{m:02d}:{s:02d}.{c:02d}] {line_text.strip()}")

        return "\n".join(lrc_lines)

    def _get_segments(
        self,
        audio_bytes: bytes,
        context_prompt: str,
        *,
        use_word_timestamps: bool,
        temperature: float,
        model_name: str | None,
        device: str | None,
        beam_size: int | None,
        best_of: int | None,
    ) -> list[dict]:
        """Returns normalized segments using Groq when available, else local Whisper."""
        if self._settings.groq_api_key:
            try:
                return self._transcribe_groq(audio_bytes, context_prompt, use_word_timestamps, temperature)
            except Exception as exc:
                logger.warning("Groq transcription failed (%s). Falling back to local Whisper.", exc)

        return self._transcribe_local(
            audio_bytes,
            context_prompt,
            model_name=model_name,
            device=device,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            use_word_timestamps=use_word_timestamps,
        )

    # ------------------------------------------------------------------- Groq

    def _transcribe_groq(
        self,
        audio_bytes: bytes,
        context_prompt: str,
        use_word_timestamps: bool,
        temperature: float,
    ) -> list[dict]:
        from groq import Groq

        # Compress to mono 16 kHz MP3 to stay well under Groq's 25 MB limit.
        compressed = self._conversion.compress_for_transcription(audio_bytes)

        client = Groq(api_key=self._settings.groq_api_key)
        granularities = ["segment", "word"] if use_word_timestamps else ["segment"]

        logger.info("Transcribing with Groq model '%s'", self._settings.groq_model)
        resp = client.audio.transcriptions.create(
            file=("audio.mp3", compressed),
            model=self._settings.groq_model,
            response_format="verbose_json",
            language=self._settings.whisper_language,
            prompt=context_prompt,
            temperature=temperature,
            timestamp_granularities=granularities,
        )

        segments_raw = self._get(resp, "segments") or []
        words_raw = self._get(resp, "words") or []

        normalized: list[dict] = []
        for seg in segments_raw:
            s_start = float(self._get(seg, "start") or 0.0)
            s_end = float(self._get(seg, "end") or s_start)
            s_text = (self._get(seg, "text") or "").strip()

            seg_words: list[dict] = []
            if use_word_timestamps:
                # Groq returns a flat words list; assign each to its segment by time.
                for w in words_raw:
                    w_start = float(self._get(w, "start") or 0.0)
                    in_range = s_start <= w_start < s_end or (s_end <= s_start and w_start >= s_start)
                    if in_range:
                        seg_words.append({"start": w_start, "word": (self._get(w, "word") or "").strip()})

            normalized.append({"start": s_start, "text": s_text, "words": seg_words})

        return normalized

    # ------------------------------------------------------------------ local

    def _transcribe_local(
        self,
        audio_bytes: bytes,
        context_prompt: str,
        *,
        model_name: str | None,
        device: str | None,
        beam_size: int | None,
        best_of: int | None,
        temperature: float,
        use_word_timestamps: bool,
    ) -> list[dict]:
        temp_path = self._settings.data_dir / f"temp-audio-{uuid.uuid4().hex}.wav"
        try:
            temp_path.write_bytes(audio_bytes)
            self._settings.whisper_download_root.mkdir(parents=True, exist_ok=True)
            model = get_whisper_model(
                model_name or self._settings.whisper_model,
                str(self._settings.whisper_download_root),
                device=device or self._settings.whisper_device,
            )

            result = model.transcribe(
                str(temp_path),
                language=self._settings.whisper_language,
                task="transcribe",
                verbose=False,
                fp16=False,
                temperature=temperature,
                beam_size=beam_size or self._settings.whisper_beam_size,
                best_of=best_of or self._settings.whisper_best_of,
                condition_on_previous_text=False,
                word_timestamps=use_word_timestamps,
                initial_prompt=context_prompt,
            )

            normalized: list[dict] = []
            for seg in result["segments"]:
                words: list[dict] = []
                if use_word_timestamps and seg.get("words"):
                    words = [{"start": w["start"], "word": w["word"]} for w in seg["words"]]
                normalized.append({"start": seg["start"], "text": seg["text"], "words": words})

            return normalized
        finally:
            temp_path.unlink(missing_ok=True)
