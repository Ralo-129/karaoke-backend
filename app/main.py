from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.dependencies import get_storage_service
from app.core.logging import configure_logging
from app.middleware.errors import add_error_handlers
from app.middleware.logging import add_logging_middleware


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    # Override temporary directory to E: drive because C: drive is 100% full (0.00 GB free).
    # This prevents Starlette's body parsing and SpooledTemporaryFile write failures.
    import tempfile
    temp_dir = settings.data_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_dir)

    # Configure torch/hub/cache directories. Prefer explicit env vars (e.g. set in
    # .env) so users can point caches to another drive (D:) with more space.
    torch_home = os.getenv("TORCH_HOME") or str(settings.data_dir / ".torch")
    xdg_cache = os.getenv("XDG_CACHE_HOME") or str(settings.data_dir / ".cache")

    # Ensure directories exist and then set env vars so downstream imports
    # (demucs/torch) write caches to these locations instead of the user's profile.
    try:
        os.makedirs(torch_home, exist_ok=True)
        os.makedirs(xdg_cache, exist_ok=True)
    except Exception:
        # Best effort; if creation fails, fall back to not overriding env vars.
        pass

    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)

    # Import routes after environment/cache dirs are configured so any imports
    # that trigger model downloads use the configured cache location.
    from app.routes import catalog, jobs, separate

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        for temp_file in settings.data_dir.glob("temp-*"):
            try:
                temp_file.unlink()
                logger.info("Limpiando archivo temporal huérfano: %s", temp_file.name)
            except Exception:
                pass

        get_storage_service().ensure_buckets()
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info("ffmpeg detected at startup: %s", ffmpeg_path)
        else:
            logger.warning(
                "ffmpeg was not found in PATH at startup. Video uploads will fail until the host installs ffmpeg."
            )
        # Optionally preload ML models into memory (controlled by PRELOAD_MODELS).
        try:
            from app.services.audio.model_manager import preload_models

            if settings.preload_models:
                preload_models(settings)
        except Exception:
            # Keep startup resilient if model preload fails or Demucs/Whisper not installed.
            pass
        yield

    app = FastAPI(title=settings.app_title, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    add_logging_middleware(app)
    add_error_handlers(app)

    app.include_router(catalog.router)
    app.include_router(separate.router)
    app.include_router(jobs.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
