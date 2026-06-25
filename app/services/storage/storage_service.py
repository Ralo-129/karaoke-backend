from __future__ import annotations

import logging
from typing import Iterable

# Supabase storage access helpers.

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self, client, uploads_bucket: str, outputs_bucket: str) -> None:
        self._client = client
        self.uploads_bucket = uploads_bucket
        self.outputs_bucket = outputs_bucket

    def ensure_buckets(self) -> None:
        for bucket in [self.uploads_bucket, self.outputs_bucket]:
            try:
                self._client.storage.get_bucket(bucket)
            except Exception:
                # Create the bucket as PUBLIC so the frontend can stream files
                # directly from Supabase public URLs without going through the
                # backend (see public_url()).
                self._client.storage.create_bucket(bucket, options={"public": True})
                logger.info("Created public storage bucket: %s", bucket)

    def upload_bytes(self, bucket: str, path: str, payload: bytes) -> None:
        # upsert=true so retries / re-processing the same job_id overwrite the
        # existing object instead of failing with a "Duplicate" error.
        self._client.storage.from_(bucket).upload(
            path, payload, file_options={"upsert": "true"}
        )

    def download_bytes(self, bucket: str, path: str) -> bytes:
        return self._client.storage.from_(bucket).download(path)

    def public_url(self, bucket: str, path: str) -> str:
        return self._client.storage.from_(bucket).get_public_url(path)

    def delete_paths(self, bucket: str, paths: Iterable[str]) -> None:
        if not paths:
            return
        self._client.storage.from_(bucket).remove(list(paths))

    def upload_original(self, job_id: str, filename: str, payload: bytes) -> str:
        file_path = f"{job_id}/{filename}"
        self.upload_bytes(self.uploads_bucket, file_path, payload)
        return file_path

    def upload_instrumental(self, job_id: str, filename: str, payload: bytes) -> str:
        file_path = f"{job_id}/{filename}"
        self.upload_bytes(self.outputs_bucket, file_path, payload)
        return file_path

    def upload_public_url(self, job_id: str, filename: str) -> str:
        """Public Supabase URL for an original upload (served without the backend)."""
        return self.public_url(self.uploads_bucket, f"{job_id}/{filename}")

    def output_public_url(self, job_id: str, filename: str) -> str:
        """Public Supabase URL for a processed output (served without the backend)."""
        return self.public_url(self.outputs_bucket, f"{job_id}/{filename}")

    def download_upload(self, job_id: str, filename: str) -> bytes:
        file_path = f"{job_id}/{filename}"
        return self.download_bytes(self.uploads_bucket, file_path)

    def download_output(self, job_id: str, filename: str) -> bytes:
        file_path = f"{job_id}/{filename}"
        return self.download_bytes(self.outputs_bucket, file_path)

    def _object_path_for(self, url: str | None, bucket: str, legacy_prefix: str) -> str | None:
        """Extract the storage object path from either a new absolute Supabase
        public URL (…/object/public/<bucket>/<path>) or a legacy relative URL
        (e.g. /uploads/<path> or /files/<path>)."""
        if not isinstance(url, str) or not url:
            return None

        marker = f"/object/public/{bucket}/"
        if marker in url:
            path = url.split(marker, 1)[1].split("?", 1)[0]
            return path or None

        if url.startswith(legacy_prefix):
            path = url.removeprefix(legacy_prefix)
            return path or None

        return None

    def delete_song_files(self, video_url: str | None, instrumental_url: str | None) -> None:
        cleanup_targets: list[tuple[str, str]] = []

        uploads_path = self._object_path_for(video_url, self.uploads_bucket, "/uploads/")
        if uploads_path:
            cleanup_targets.append((self.uploads_bucket, uploads_path))

        outputs_path = self._object_path_for(instrumental_url, self.outputs_bucket, "/files/")
        if outputs_path:
            cleanup_targets.append((self.outputs_bucket, outputs_path))

        for bucket, path in cleanup_targets:
            try:
                self.delete_paths(bucket, [path])
            except Exception as exc:
                logger.warning("Cleanup failed for %s/%s: %s", bucket, path, exc)
