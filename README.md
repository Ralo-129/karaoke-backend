# karaoke-backend

FastAPI service that accepts audio/video uploads, extracts the lyrics with
Whisper, separates the instrumental with Demucs, and stores everything in
Supabase (Postgres + Storage).

## Requirements

- Python 3.10+
- ffmpeg installed on the host (needed to extract audio from video uploads)

## Environment variables

Create a `.env` (see `.env.example`):

| Variable | Required | Default | Notes |
|---|---|---|---|
| `SUPABASE_URL` | yes | — | Supabase project URL |
| `SUPABASE_SECRET` | yes* | — | Service role key (bypasses RLS). Preferred. |
| `SUPABASE_KEY` | yes* | — | Anon key (used only if no secret) |
| `ADMIN_TOKEN` | recommended | — | Shared secret required to delete songs / reset the catalog. If unset, those endpoints return 503 (fail closed). |
| `GROQ_API_KEY` | recommended | — | If set, transcription uses Groq's hosted Whisper Large v3 (faster, frees RAM). Empty = local Whisper. |
| `GROQ_WHISPER_MODEL` | no | `whisper-large-v3` | `whisper-large-v3` (max accuracy) or `whisper-large-v3-turbo` (faster). |
| `DEMUCS_MODEL` | no | `htdemucs` | 4-stem model (lighter). Avoid `htdemucs_6s` (heavier). |
| `WHISPER_MODEL` | no | `tiny` | Local fallback model (only used when `GROQ_API_KEY` is empty). |
| `PRELOAD_MODELS` | no | `false` | Preload models at startup |

\* `SUPABASE_URL` plus one of `SUPABASE_SECRET` / `SUPABASE_KEY` is required.

## Local run

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```
docker build -t karaoke-backend .
docker run --env-file .env -p 8000:8000 karaoke-backend
```

First run, create the DB table + buckets: run `SUPABASE_SETUP.sql` in the Supabase
SQL editor (or `python setup_supabase.py`).

## API

- `GET /health` -> `{ "status": "ok" }`
- `POST /separate` -> upload (supports chunked upload); kicks off processing
- `GET /catalog` / `GET /catalog/{id}` -> read catalog (public)
- `POST /catalog/{id}/separate` -> generate instrumental on demand
- `DELETE /catalog/{id}` / `DELETE /catalog` -> **requires `X-Admin-Token` header**
- `GET /jobs/{job_id}` -> processing status (in-memory)
- `GET /files/...` & `GET /uploads/...` -> legacy file streaming (kept for
  backward compatibility; new records use direct Supabase public URLs)

## Architecture & hosting notes

Processing (Whisper + Demucs) is CPU/RAM heavy; serving songs is not. They are
decoupled:

- **Processing** (uploads) needs this backend running. Run it locally (Docker /
  uvicorn) only while adding songs. Avoid tiny free tiers (e.g. Azure F1) — they
  don't have the RAM/CPU and time out.
- **Playback**: processed files get **public Supabase URLs**, so audio/video
  stream straight from Supabase (CDN) without the backend. The frontend can be
  hosted free (Vercel).

### Lowering resource use

- **Transcription via Groq** (set `GROQ_API_KEY`): offloads Whisper to Groq's API,
  freeing local RAM/CPU and improving lyric accuracy. Falls back to local Whisper
  if the key is missing or the call fails. Demucs still runs locally.
- Default Demucs model is `htdemucs` (4 stems), not `htdemucs_6s` (6 stems).
- The `rapido` profile uses `shifts=0` / low `overlap` and disables Whisper word
  timestamps for the fastest, lightest processing.
- The vocal stem is no longer computed (only the instrumental is needed).
- Audio is extracted to WAV once and reused by both Whisper and Demucs.
- Pasting an already-timed **LRC** in the upload form skips Whisper entirely.

## Notes

- Video uploads need ffmpeg to extract audio before Whisper and Demucs run.
