from __future__ import annotations

import logging
import uuid

from app.core.config import Settings
from app.services.audio.conversion_service import AudioConversionService
from app.services.audio.model_manager import get_whisper_model

# Whisper transcription utilities. Uses Groq's hosted Whisper Large v3 when a
# GROQ_API_KEY is configured (faster + frees local RAM), falling back to the
# local openai-whisper model otherwise (or if the Groq call fails).

logger = logging.getLogger(__name__)

_YOUTUBE_ARTIFACTS = {
    "gracias por ver",
    "suscribete",
    "suscríbete",
    "like y comparte",
    "dale like",
    "no olvides suscribirte",
    "comparte el video",
    "activa la campana",
    "activa las notificaciones",
    "subtitulado por",
    "subtítulos por",
    "nos vemos",
}


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
        try:
            if self._settings.groq_api_key:
                try:
                    # ⚠️  NO ALARGAR ESTE PROMPT — PROBLEMA CONOCIDO CON GROQ ⚠️
                    #
                    # Groq Whisper trata el parámetro `prompt` como la transcripción
                    # de audio ANTERIOR al clip actual. Si el prompt contiene frases
                    # largas (título, artista, instrucciones, letra de referencia),
                    # el modelo las "escucha" en el audio y las inserta como letras
                    # fantasma en el LRC de salida. Ejemplos reales observados:
                    #   - "No resumir ni traducir" → aparece en 02:15 del LRC
                    #   - "Titulo: Mala Sombra." → aparece manglado en 01:21
                    #   - letra de referencia en el prompt → timestamps comprimidos
                    #     a 70ms para 10 palabras (karaoke totalmente desincronizado)
                    #
                    # La solución es mantener el prompt de Groq en ≤5 palabras,
                    # solo para fijar el idioma. El Whisper local usa `initial_prompt`
                    # de forma diferente y SÍ puede recibir hint_lyrics sin este bug.
                    groq_prompt = "Letra en español."
                    return self._transcribe_groq(
                        audio_bytes, groq_prompt, use_word_timestamps, temperature
                    )
                except Exception as exc:
                    logger.warning(
                        "Groq transcription failed (%s). Falling back to local Whisper.", exc
                    )

            local_prompt = self._build_local_prompt(title, artist, hint_lyrics)
            return self._transcribe_local(
                audio_bytes,
                local_prompt,
                model_name=model_name,
                device=device,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                use_word_timestamps=use_word_timestamps,
            )
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return ""

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _build_local_prompt(self, title: str, artist: str, hint_lyrics: str | None) -> str:
        parts = ["Transcripcion de letra en español."]
        if title:
            parts.append(f"Titulo: {title}.")
        if artist:
            parts.append(f"Artista: {artist}.")
        parts.append("Solo transcribir lo que se canta. No resumir ni traducir.")
        if hint_lyrics:
            snippet = " ".join(hint_lyrics.split())[:600]
            parts.append(f"Referencia: {snippet}")
        return " ".join(parts)

    @staticmethod
    def _is_artifact(text: str) -> bool:
        clean = text.lower().strip().rstrip(".,!¡¿?")
        return any(artifact in clean for artifact in _YOUTUBE_ARTIFACTS)

    @staticmethod
    def _is_valid_segment(seg_words: list[dict]) -> bool:
        if len(seg_words) < 2:
            return True
        starts = [w["start"] for w in seg_words]
        backwards = sum(1 for a, b in zip(starts, starts[1:]) if b < a)
        if (backwards / len(starts)) >= 0.3:
            return False
        # Ultracompressed timestamps (>3 words in <0.5s) indicate hallucination
        if len(seg_words) > 3 and (max(starts) - min(starts)) < 0.5:
            return False
        return True

    def _build_lrc(self, segments: list[dict], use_word_timestamps: bool) -> str:
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

            clean = line_text.strip()
            if clean and not self._is_artifact(clean):
                lrc_lines.append(f"[{m:02d}:{s:02d}.{c:02d}] {clean}")

        return "\n".join(lrc_lines)

    # ------------------------------------------------------------------- Groq

    def _transcribe_groq(
        self,
        audio_bytes: bytes,
        context_prompt: str,
        use_word_timestamps: bool,
        temperature: float,
    ) -> str:
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

        # Sort words by timestamp once so segment assignment is stable.
        words_sorted = sorted(words_raw, key=lambda w: float(self._get(w, "start") or 0.0))

        normalized: list[dict] = []
        for seg in segments_raw:
            s_start = float(self._get(seg, "start") or 0.0)
            s_end = float(self._get(seg, "end") or s_start)
            s_text = (self._get(seg, "text") or "").strip()

            seg_words: list[dict] = []
            if use_word_timestamps:
                for w in words_sorted:
                    w_start = float(self._get(w, "start") or 0.0)
                    in_range = s_start <= w_start < s_end or (s_end <= s_start and w_start >= s_start)
                    if in_range:
                        seg_words.append({"start": w_start, "word": (self._get(w, "word") or "").strip()})

            if not self._is_valid_segment(seg_words):
                logger.debug("Descartando segmento con timestamps inválidos en t=%.2f: %s", s_start, s_text)
                continue

            normalized.append({"start": s_start, "text": s_text, "words": seg_words})

        return self._build_lrc(normalized, use_word_timestamps)

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
    ) -> str:
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

            return self._build_lrc(normalized, use_word_timestamps)
        finally:
            temp_path.unlink(missing_ok=True)
