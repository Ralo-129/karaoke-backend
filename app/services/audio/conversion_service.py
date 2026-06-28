from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.audio import convert_audio

from app.common.formatters import format_duration
from app.core.config import Settings

# Audio conversion and probing utilities.


class AudioConversionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def convert_to_wav(self, input_path: Path, output_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed (code {result.returncode}):\n{result.stderr[-2000:]}"
            )

    def load_audio_for_demucs(
        self, wav_path: Path, target_channels: int, target_samplerate: int
    ) -> torch.Tensor:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        wav = convert_audio(wav, sr, target_samplerate, target_channels)
        return wav

    def audio_to_mp3_bytes(self, audio_np: np.ndarray, sr: int, quality: int = 2) -> bytes:
        temp_wav = self._settings.data_dir / f"temp-mp3-input-{uuid.uuid4().hex}.wav"
        temp_mp3 = self._settings.data_dir / f"temp-mp3-output-{uuid.uuid4().hex}.mp3"
        try:
            sf.write(str(temp_wav), audio_np, sr, format="WAV")
            mp3_quality = str(max(0, min(9, quality)))
            cmd = ["ffmpeg", "-y", "-i", str(temp_wav), "-codec:a", "libmp3lame", "-qscale:a", mp3_quality, str(temp_mp3)]
            subprocess.run(cmd, capture_output=True, check=True)
            return temp_mp3.read_bytes()
        finally:
            temp_wav.unlink(missing_ok=True)
            temp_mp3.unlink(missing_ok=True)

    def compress_for_transcription(self, wav_bytes: bytes) -> bytes:
        """Compress audio to mono 16 kHz MP3 for the Groq Speech-to-Text API.

        Whisper internally downsamples to 16 kHz mono anyway, so this keeps full
        transcription quality while shrinking the file to ~1 MB/min — well under
        Groq's 25 MB upload limit and much faster to send.
        """
        temp_in = self._settings.data_dir / f"temp-stt-in-{uuid.uuid4().hex}.wav"
        temp_out = self._settings.data_dir / f"temp-stt-out-{uuid.uuid4().hex}.mp3"
        try:
            temp_in.write_bytes(wav_bytes)
            cmd = [
                "ffmpeg", "-y", "-i", str(temp_in),
                "-ac", "1", "-ar", "16000",
                "-codec:a", "libmp3lame", "-qscale:a", "5",
                str(temp_out),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg compress failed (code {result.returncode}):\n{result.stderr[-2000:]}"
                )
            return temp_out.read_bytes()
        finally:
            temp_in.unlink(missing_ok=True)
            temp_out.unlink(missing_ok=True)

    def probe_duration_from_path(self, path: Path) -> str:
        try:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return format_duration(float(result.stdout.strip()))
        except Exception:
            return "--:--"

    def probe_duration_label(self, file_bytes: bytes, filename: str) -> str:
        temp_path = self._settings.data_dir / f"temp-{uuid.uuid4().hex}-{filename}"
        try:
            temp_path.write_bytes(file_bytes)
            return self.probe_duration_from_path(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
