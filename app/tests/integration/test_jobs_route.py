from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.song import SongPublic


@pytest.fixture
def mock_jobs_svc():
    return MagicMock()


@pytest.fixture
def jobs_client(mock_jobs_svc, mock_startup_storage, settings_env):
    from app.main import app
    with (
        patch("app.main.get_storage_service", return_value=mock_startup_storage),
        patch("app.routes.jobs.get_jobs_service", return_value=mock_jobs_svc),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_jobs_svc


def _make_song_public(**kwargs) -> SongPublic:
    defaults = {"id": "abc123", "title": "Song", "artist": "Artist"}
    return SongPublic(**{**defaults, **kwargs})


class TestGetJobStatus:
    def test_returns_processing_status(self, jobs_client):
        client, svc = jobs_client
        svc.get_status.return_value = {
            "job_id": "job1",
            "status": "processing",
            "progress": 45,
            "message": "Transcribiendo...",
            "song": None,
        }
        resp = client.get("/jobs/job1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processing"
        assert body["progress"] == 45
        assert body["song"] is None

    def test_returns_completed_status_with_song(self, jobs_client):
        client, svc = jobs_client
        song = _make_song_public()
        svc.get_status.return_value = {
            "job_id": "job1",
            "status": "completed",
            "progress": 100,
            "message": "Completado",
            "song": song.model_dump(),
        }
        resp = client.get("/jobs/job1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["song"] is not None
        assert body["song"]["title"] == "Song"

    def test_returns_error_status(self, jobs_client):
        client, svc = jobs_client
        svc.get_status.return_value = {
            "job_id": "job1",
            "status": "error",
            "progress": 0,
            "message": "ffmpeg not found",
            "song": None,
        }
        resp = client.get("/jobs/job1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "ffmpeg" in body["message"]
