from __future__ import annotations

from pathlib import Path

import numpy as np
from demucs.apply import apply_model

from app.services.audio.model_manager import get_demucs_model

from app.core.config import Settings
from app.services.audio.conversion_service import AudioConversionService

# Demucs-based separation logic.


class AudioSeparationService:
    def __init__(self, settings: Settings, conversion_service: AudioConversionService) -> None:
        self._settings = settings
        self._conversion = conversion_service

    def separate(
        self,
        file_bytes: bytes,
        original_filename: str,
        job_id: str,
        *,
        model_name: str | None = None,
        device: str | None = None,
        mp3_quality: int = 2,
        shifts: int = 1,
        overlap: float = 0.25,
        already_wav: bool = False,
    ) -> bytes:
        """Separate the instrumental (everything except vocals) and return MP3 bytes.

        - ``already_wav``: when True, ``file_bytes`` is treated as a WAV that was
          already extracted upstream, so we skip the ffmpeg re-conversion.
        - ``shifts`` / ``overlap``: control Demucs cost (lower = faster / less RAM).

        The vocal stem is intentionally NOT produced: we only need the karaoke
        instrumental, so computing/encoding vocals would just waste CPU and memory.
        """
        model = get_demucs_model(model_name or self._settings.demucs_model, device or self._settings.demucs_device)
        original_suffix = Path(original_filename).suffix.lower() or ".bin"
        temp_input = self._settings.data_dir / f"temp-input-{job_id}{original_suffix}"
        temp_wav = self._settings.data_dir / f"temp-input-{job_id}.wav"

        try:
            if already_wav:
                temp_wav.write_bytes(file_bytes)
            else:
                temp_input.write_bytes(file_bytes)
                try:
                    self._conversion.convert_to_wav(temp_input, temp_wav)
                except Exception:
                    temp_wav = temp_input

            mix = (
                self._conversion.load_audio_for_demucs(
                    temp_wav, model.audio_channels, model.samplerate
                )
                .unsqueeze(0)
            )
            sources = apply_model(
                model,
                mix,
                device=self._settings.demucs_device,
                shifts=shifts,
                overlap=overlap,
            )

            try:
                sources_np = sources.detach().cpu().numpy()
            except Exception:
                sources_np = sources.cpu().numpy()

            batch_idx = 0
            non_vocals = []
            for idx, name in enumerate(model.sources):
                if name == "vocals":
                    continue
                non_vocals.append(sources_np[batch_idx, idx])

            if non_vocals:
                instrumental_np = np.sum(np.stack(non_vocals, axis=0), axis=0)
            else:
                if "vocals" in model.sources:
                    vocals_idx = model.sources.index("vocals")
                    vocals_np = sources_np[batch_idx, vocals_idx]
                    try:
                        mix_np = mix.detach().cpu().numpy()[batch_idx]
                    except Exception:
                        mix_np = mix.cpu().numpy()[batch_idx]
                    instrumental_np = mix_np - vocals_np
                else:
                    try:
                        instrumental_np = mix.detach().cpu().numpy()[batch_idx]
                    except Exception:
                        instrumental_np = mix.cpu().numpy()[batch_idx]

            max_val = float(np.max(np.abs(instrumental_np))) if instrumental_np.size else 0.0
            if max_val > 1.0:
                instrumental_np = instrumental_np / max_val

            # Convert instrumental to MP3 bytes to stay under Supabase limits
            return self._conversion.audio_to_mp3_bytes(
                instrumental_np.T,
                model.samplerate,
                quality=mp3_quality,
            )
        finally:
            temp_input.unlink(missing_ok=True)
            if temp_wav != temp_input:
                temp_wav.unlink(missing_ok=True)
