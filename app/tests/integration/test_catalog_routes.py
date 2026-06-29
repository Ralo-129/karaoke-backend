from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.song import SongRecord


ADMIN_TOKEN = "test-admin-token"


def _make_song(**kwargs) -> SongRecord:
    defaults = {"job_id": "abc123", "title": "Test Song", "artist": "Artista"}
    return SongRecord(**{**defaults, **kwargs})


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def catalog_client(mock_repo, mock_storage, mock_startup_storage, settings_env):
    from app.main import app
    with (
        patch("app.main.get_storage_service", return_value=mock_startup_storage),
        patch("app.routes.catalog.get_songs_repository", return_value=mock_repo),
        patch("app.routes.catalog.get_storage_service", return_value=mock_storage),
        patch("app.routes.catalog.get_jobs_service", return_value=MagicMock()),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_repo, mock_storage


class TestListCatalog:
    def test_returns_empty_list(self, catalog_client):
        client, repo, _ = catalog_client
        repo.list_songs.return_value = []
        resp = client.get("/catalog")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_songs(self, catalog_client):
        client, repo, _ = catalog_client
        repo.list_songs.return_value = [_make_song()]
        resp = client.get("/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["title"] == "Test Song"
        assert data[0]["id"] == "abc123"

    def test_absorbs_repo_exception(self, catalog_client):
        client, repo, _ = catalog_client
        repo.list_songs.side_effect = Exception("DB down")
        resp = client.get("/catalog")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetCatalogSong:
    def test_returns_song_when_found(self, catalog_client):
        client, repo, _ = catalog_client
        repo.get_song.return_value = _make_song()
        resp = client.get("/catalog/abc123")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Test Song"

    def test_returns_404_when_not_found(self, catalog_client):
        client, repo, _ = catalog_client
        repo.get_song.return_value = None
        resp = client.get("/catalog/notexist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestDeleteCatalogSong:
    def test_requires_admin_token(self, catalog_client):
        client, _, _ = catalog_client
        resp = client.delete("/catalog/abc123")
        assert resp.status_code == 401

    def test_rejects_wrong_token(self, catalog_client):
        client, _, _ = catalog_client
        resp = client.delete("/catalog/abc123", headers={"X-Admin-Token": "wrong"})
        assert resp.status_code == 401

    def test_deletes_song_with_correct_token(self, catalog_client):
        client, repo, storage = catalog_client
        song = _make_song(video_url="http://v", instrumental_url="http://i")
        repo.delete_song.return_value = song
        resp = client.delete("/catalog/abc123", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        storage.delete_song_files.assert_called_once_with("http://v", "http://i")

    def test_returns_404_if_song_not_found(self, catalog_client):
        client, repo, _ = catalog_client
        repo.delete_song.return_value = None
        resp = client.delete("/catalog/notexist", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert resp.status_code == 404


class TestResetCatalog:
    def test_requires_admin_token(self, catalog_client):
        client, _, _ = catalog_client
        resp = client.delete("/catalog")
        assert resp.status_code == 401

    def test_clears_catalog_with_correct_token(self, catalog_client):
        client, repo, _ = catalog_client
        resp = client.delete("/catalog", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        repo.reset_catalog.assert_called_once()
