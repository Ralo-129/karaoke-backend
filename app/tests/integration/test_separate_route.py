from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_jobs_svc():
    svc = MagicMock()
    svc.handle_upload = AsyncMock(return_value={
        "job_id": "test-job-id",
        "status": "processing",
        "message": "Upload complete, processing started.",
    })
    return svc


@pytest.fixture
def separate_client(mock_jobs_svc, mock_startup_storage, settings_env):
    from app.main import app
    with (
        patch("app.main.get_storage_service", return_value=mock_startup_storage),
        patch("app.routes.separate.get_jobs_service", return_value=mock_jobs_svc),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_jobs_svc


class TestSeparateEndpoint:
    def test_missing_file_returns_400(self, separate_client):
        client, _ = separate_client
        resp = client.post("/separate")
        # The custom error handler in middleware/errors.py maps RequestValidationError → 400
        assert resp.status_code == 400

    def test_file_too_large_returns_413(self, separate_client, monkeypatch):
        client, _ = separate_client
        monkeypatch.setenv("MAX_UPLOAD_MB", "1")
        from app.core.config import get_settings
        get_settings.cache_clear()

        big_content = b"x" * (2 * 1024 * 1024)
        resp = client.post(
            "/separate",
            files={"file": ("big.mp3", big_content, "audio/mpeg")},
        )
        assert resp.status_code == 413

    def test_valid_upload_calls_service_and_returns_body(self, separate_client):
        client, svc = separate_client
        resp = client.post(
            "/separate",
            files={"file": ("song.mp3", b"fake-audio-data", "audio/mpeg")},
            data={"title": "Mi canción", "artist": "Artista"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "test-job-id"
        assert body["status"] == "processing"
        svc.handle_upload.assert_called_once()

    def test_chunk_upload_returns_chunk_received(self, separate_client):
        client, svc = separate_client
        svc.handle_upload = AsyncMock(return_value={
            "job_id": "test-job-id",
            "status": "chunk_received",
            "chunk": 0,
            "total": 3,
        })
        resp = client.post(
            "/separate",
            files={"file": ("song.mp3.chunk_0", b"chunk-data", "application/octet-stream")},
            data={"job_id": "test-job-id", "chunk_index": "0", "total_chunks": "3"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "chunk_received"
