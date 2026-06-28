from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException

from app.core.config import get_settings
from app.core.dependencies import get_songs_repository, get_storage_service, get_jobs_service
from app.models.song import to_public

# Catalog endpoints.

logger = logging.getLogger(__name__)
router = APIRouter()


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Guards destructive endpoints. Requires a matching X-Admin-Token header.

    Fails closed: if ADMIN_TOKEN is not configured on the server, destructive
    operations are disabled entirely (503) instead of being left wide open.
    """
    expected = get_settings().admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="Admin token not configured on server.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized.")


@router.get("/catalog")
def list_catalog() -> list[dict[str, Any]]:
    try:
        songs = get_songs_repository().list_songs()
        return [to_public(song).model_dump() for song in songs]
    except Exception as exc:
        logger.error("Catalog error: %s", exc)
        return []


@router.get("/catalog/{song_id}")
def get_catalog_song(song_id: str) -> dict[str, Any]:
    song = get_songs_repository().get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")
    return to_public(song).model_dump()


@router.post("/catalog/{song_id}/separate")
async def separate_song_instrumental(
    song_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    song = get_songs_repository().get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")
    if song.instrumental_url:
        return {"status": "ok", "instrumental_url": song.instrumental_url}

    background_tasks.add_task(get_jobs_service().separate_instrumental_on_demand, song_id)
    return {"status": "processing", "job_id": song_id, "message": "Separación iniciada."}


@router.delete("/catalog/{song_id}")
def delete_catalog_song(song_id: str, _: None = Depends(require_admin)) -> dict[str, str]:
    song = get_songs_repository().delete_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")

    get_storage_service().delete_song_files(song.video_url, song.instrumental_url)
    return {"status": "ok", "message": "Song deleted"}


@router.delete("/catalog")
def reset_catalog(_: None = Depends(require_admin)) -> dict[str, str]:
    try:
        get_songs_repository().reset_catalog()
        return {"status": "ok", "message": "Catalog cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error clearing catalog: {str(exc)}")
