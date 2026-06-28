#!/usr/bin/env python3
"""Migra los archivos de Supabase Storage a Cloudflare R2 y actualiza las URLs
en la tabla `songs`. La base de datos sigue en Supabase; solo se mueven los
archivos. Idempotente: salta lo que ya tiene URL de R2.

Uso (desde karaoke-backend, con el .env configurado):
    python migrate_to_r2.py
"""

from __future__ import annotations

import sys
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv()

from app.core.config import get_settings  # noqa: E402
from app.services.storage.r2_storage_service import R2StorageService  # noqa: E402
from supabase import create_client  # noqa: E402


def parse_supabase_path(url: str | None) -> tuple[str, str] | None:
    """Returns (bucket, path) for a Supabase storage URL, or None.

    path is "<job_id>/<filename>"."""
    if not url:
        return None
    marker = "/object/public/"
    if marker in url:
        tail = unquote(url.split(marker, 1)[1].split("?", 1)[0])  # "<bucket>/<job>/<file>"
        parts = tail.split("/", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else None
    # Legacy relative URLs.
    if url.startswith("/files/"):
        return ("outputs", unquote(url[len("/files/"):]))
    if url.startswith("/uploads/"):
        return ("uploads", unquote(url[len("/uploads/"):]))
    return None


def main() -> None:
    settings = get_settings()
    if not (settings.r2_endpoint and settings.r2_access_key_id and settings.r2_secret_access_key):
        print("[ERROR] R2 no esta configurado en el .env (R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY).")
        sys.exit(1)

    sb = create_client(settings.supabase_url, settings.supabase_secret or settings.supabase_key)
    r2 = R2StorageService(settings)
    public_base = settings.r2_public_base_url

    rows = sb.table("songs").select("*").execute().data or []
    print(f"[INFO] {len(rows)} canciones en el catalogo.")

    moved = skipped = failed = 0

    for row in rows:
        job_id = row.get("job_id")
        updates: dict[str, str] = {}

        for field in ("video_url", "instrumental_url"):
            url = row.get(field)
            if not url:
                continue
            if public_base and url.startswith(public_base):
                skipped += 1
                continue  # ya esta en R2

            parsed = parse_supabase_path(url)
            if not parsed:
                continue
            bucket, path = parsed
            seg = path.split("/", 1)
            if len(seg) != 2:
                continue
            job_seg, filename = seg

            try:
                data = sb.storage.from_(bucket).download(path)
                if bucket == "outputs":
                    r2.upload_instrumental(job_seg, filename, data)
                    new_url = r2.output_public_url(job_seg, filename)
                else:
                    r2.upload_original(job_seg, filename, data)
                    new_url = r2.upload_public_url(job_seg, filename)
                updates[field] = new_url
                moved += 1
                print(f"[OK] {field} {job_id} -> {new_url}")
            except Exception as exc:
                failed += 1
                print(f"[FALLO] {field} {job_id}: {exc}")

        if updates:
            sb.table("songs").update(updates).eq("job_id", job_id).execute()

    print(f"\n[LISTO] Movidos: {moved} | Ya en R2: {skipped} | Fallidos: {failed}")
    if moved:
        print("Las URLs en la tabla 'songs' se actualizaron a R2. Prueba reproducir una cancion.")


if __name__ == "__main__":
    main()
