from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Centralized environment and path configuration.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_root: Path
    data_dir: Path
    chunks_dir: Path
    uploads_bucket: str
    outputs_bucket: str
    supabase_url: str
    supabase_key: str
    supabase_secret: str
    admin_token: str
    demucs_model: str
    demucs_device: str
    whisper_model: str
    whisper_device: str
    whisper_language: str
    whisper_beam_size: int
    whisper_best_of: int
    whisper_download_root: Path
    preload_models: bool
    groq_api_key: str
    groq_model: str
    app_title: str = "karaoke-backend"


def _require(value: str, name: str) -> str:
    if not value:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY or SUPABASE_SECRET must be set in environment variables"
        )
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _ensure_directories(settings: Settings) -> None:
    settings.data_dir.mkdir(exist_ok=True)
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    settings.whisper_download_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    app_root = Path(__file__).resolve().parents[1]
    data_dir = app_root / "data"
    chunks_dir = data_dir / "chunks"

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    supabase_secret = os.getenv("SUPABASE_SECRET", "").strip()

    if not supabase_url or (not supabase_key and not supabase_secret):
        _require("", "SUPABASE_URL")

    uploads_bucket = os.getenv("UPLOADS_BUCKET", "uploads").strip() or "uploads"
    outputs_bucket = os.getenv("OUTPUTS_BUCKET", "outputs").strip() or "outputs"

    # Shared secret required for destructive endpoints (delete song / reset catalog).
    admin_token = os.getenv("ADMIN_TOKEN", "").strip()

    # Default to the lighter 4-stem model. "htdemucs_6s" (6 stems) is heavier and
    # unnecessary for karaoke (we only need "everything except vocals").
    demucs_model = os.getenv("DEMUCS_MODEL", "htdemucs").strip() or "htdemucs"
    demucs_device = os.getenv("DEMUCS_DEVICE", "cpu").strip() or "cpu"

    # Prefer a smaller Whisper model by default for low-memory deploys.
    whisper_model = os.getenv("WHISPER_MODEL", "tiny").strip() or "tiny"
    whisper_device = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
    whisper_language = os.getenv("WHISPER_LANGUAGE", "es").strip() or "es"
    whisper_beam_size = _int_env("WHISPER_BEAM_SIZE", 5)
    whisper_best_of = _int_env("WHISPER_BEST_OF", 5)
    whisper_download_root = Path(
        os.getenv("WHISPER_DOWNLOAD_ROOT", str(data_dir / "whisper-models"))
    )
    preload_models = _bool_env("PRELOAD_MODELS", False)

    # Groq Speech-to-Text. If GROQ_API_KEY is set, transcription uses Groq's
    # hosted Whisper Large v3 (faster + frees local RAM); otherwise it falls
    # back to the local openai-whisper model.
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    # Default to the most accurate model (best lyrics). Since transcription is a
    # remote service now, there's no local performance cost to using it.
    groq_model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3").strip() or "whisper-large-v3"

    settings = Settings(
        app_root=app_root,
        data_dir=data_dir,
        chunks_dir=chunks_dir,
        uploads_bucket=uploads_bucket,
        outputs_bucket=outputs_bucket,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        supabase_secret=supabase_secret,
        admin_token=admin_token,
        demucs_model=demucs_model,
        demucs_device=demucs_device,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_language=whisper_language,
        whisper_beam_size=whisper_beam_size,
        whisper_best_of=whisper_best_of,
        whisper_download_root=whisper_download_root,
        preload_models=preload_models,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
    )
    _ensure_directories(settings)
    return settings
