from __future__ import annotations

import logging
from urllib.parse import quote, unquote

import boto3
from botocore.config import Config

from app.core.config import Settings

# Cloudflare R2 (S3-compatible) storage backend. Same interface as the Supabase
# StorageService, but files live in a single R2 bucket under "uploads/" and
# "outputs/" prefixes and are served from the public r2.dev base URL (free egress).

logger = logging.getLogger(__name__)

_UPLOADS_PREFIX = "uploads"
_OUTPUTS_PREFIX = "outputs"

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def _content_type(filename: str) -> str:
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


class R2StorageService:
    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.r2_bucket
        self._public_base = settings.r2_public_base_url.rstrip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    # ----------------------------------------------------------------- helpers

    def _key(self, prefix: str, job_id: str, filename: str) -> str:
        return f"{prefix}/{job_id}/{filename}"

    def _public_url(self, key: str) -> str:
        return f"{self._public_base}/{quote(key, safe='/')}"

    def _put(self, key: str, payload: bytes, filename: str) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=payload,
            ContentType=_content_type(filename),
        )

    def _get(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"].read()

    # ------------------------------------------------------------- lifecycle

    def ensure_buckets(self) -> None:
        # The bucket is created by the user in the Cloudflare dashboard; just
        # verify access (best effort) so startup logs are clear.
        try:
            self._client.head_bucket(Bucket=self._bucket)
            logger.info("R2 storage ready: bucket '%s'", self._bucket)
        except Exception as exc:
            logger.warning("Could not verify R2 bucket '%s': %s", self._bucket, exc)

    # --------------------------------------------------------------- uploads

    def upload_original(self, job_id: str, filename: str, payload: bytes) -> str:
        key = self._key(_UPLOADS_PREFIX, job_id, filename)
        self._put(key, payload, filename)
        return key

    def upload_instrumental(self, job_id: str, filename: str, payload: bytes) -> str:
        key = self._key(_OUTPUTS_PREFIX, job_id, filename)
        self._put(key, payload, filename)
        return key

    def upload_public_url(self, job_id: str, filename: str) -> str:
        return self._public_url(self._key(_UPLOADS_PREFIX, job_id, filename))

    def output_public_url(self, job_id: str, filename: str) -> str:
        return self._public_url(self._key(_OUTPUTS_PREFIX, job_id, filename))

    # ------------------------------------------------------------- downloads

    def download_upload(self, job_id: str, filename: str) -> bytes:
        return self._get(self._key(_UPLOADS_PREFIX, job_id, filename))

    def download_output(self, job_id: str, filename: str) -> bytes:
        return self._get(self._key(_OUTPUTS_PREFIX, job_id, filename))

    # ---------------------------------------------------------------- delete

    def _key_from_url(self, url: str | None) -> str | None:
        if not isinstance(url, str) or not url:
            return None
        if self._public_base and url.startswith(self._public_base):
            return unquote(url[len(self._public_base):].lstrip("/")) or None
        # Legacy Supabase public URL: .../object/public/<bucket>/<path>
        marker = "/object/public/"
        if marker in url:
            tail = url.split(marker, 1)[1].split("?", 1)[0]
            return unquote(tail) or None  # e.g. "uploads/<job>/<file>"
        return None

    def delete_song_files(self, video_url: str | None, instrumental_url: str | None) -> None:
        for url in (video_url, instrumental_url):
            key = self._key_from_url(url)
            if not key:
                continue
            try:
                self._client.delete_object(Bucket=self._bucket, Key=key)
            except Exception as exc:
                logger.warning("R2 cleanup failed for %s: %s", key, exc)
