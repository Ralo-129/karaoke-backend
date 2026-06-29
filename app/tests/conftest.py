from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.repositories.jobs_repository import JobsRepository
from app.repositories.songs_repository import SongsRepository
from app.services.jobs.jobs_service import JobsService
from app.services.storage.storage_service import StorageService


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    """Dummy env vars + clear all lru_caches so each test starts clean."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-anon-key")
    monkeypatch.setenv("SUPABASE_SECRET", "")
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("R2_ENDPOINT", "")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "")

    from app.core.config import get_settings
    from app.core.dependencies import (
        get_audio_conversion_service,
        get_audio_separation_service,
        get_database_service,
        get_jobs_repository,
        get_jobs_service,
        get_songs_repository,
        get_storage_service,
        get_supabase_client,
        get_transcription_service,
    )

    caches = [
        get_settings, get_supabase_client, get_storage_service,
        get_database_service, get_songs_repository, get_jobs_repository,
        get_audio_conversion_service, get_audio_separation_service,
        get_transcription_service, get_jobs_service,
    ]
    for fn in caches:
        fn.cache_clear()

    yield

    for fn in caches:
        fn.cache_clear()


@pytest.fixture
def mock_songs_repo():
    return MagicMock(spec=SongsRepository)


@pytest.fixture
def mock_jobs_service():
    return MagicMock(spec=JobsService)


@pytest.fixture
def mock_startup_storage():
    """MagicMock for StorageService that prevents Supabase calls during app startup."""
    return MagicMock(spec=StorageService)
