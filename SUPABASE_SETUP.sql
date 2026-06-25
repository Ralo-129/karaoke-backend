-- Supabase Songs Database Setup
-- Run this in Supabase SQL Editor: https://app.supabase.com → SQL Editor

-- Create songs table with UUID, job_id, metadata, and timestamps
CREATE TABLE IF NOT EXISTS songs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    artist TEXT,
    bpm INTEGER DEFAULT 0,
    duration TEXT,
    lrc_preview TEXT,
    lrc TEXT,
    tags TEXT[] DEFAULT '{"subido"}',
    video_url TEXT,
    instrumental_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_songs_job_id ON songs(job_id);
CREATE INDEX IF NOT EXISTS idx_songs_created_at_desc ON songs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist);

-- Enable Row Level Security
ALTER TABLE songs ENABLE ROW LEVEL SECURITY;

-- Policies
-- 1. Everyone can read all songs (the catalog is meant to be public).
DROP POLICY IF EXISTS "Public read" ON songs;
CREATE POLICY "Public read" ON songs
    FOR SELECT USING (true);

-- 2/3. Writes and deletes are intentionally NOT exposed to anon/authenticated
-- roles. The backend connects with the Supabase SERVICE ROLE key, which bypasses
-- RLS, so inserts/updates/deletes only happen through the backend (which also
-- guards destructive endpoints with X-Admin-Token). This prevents anyone with
-- the public URL from wiping the catalog directly against Supabase.
-- Drop the old wide-open policies if they exist from a previous setup:
DROP POLICY IF EXISTS "Authenticated write" ON songs;
DROP POLICY IF EXISTS "Authenticated delete" ON songs;

-- Create storage buckets (can also be done via UI or API on first upload)
-- These may fail with 403 if bucket already exists - that's OK
INSERT INTO storage.buckets (id, name, public)
VALUES ('uploads', 'uploads', true)
ON CONFLICT DO NOTHING;

INSERT INTO storage.buckets (id, name, public)
VALUES ('outputs', 'outputs', true)
ON CONFLICT DO NOTHING;
