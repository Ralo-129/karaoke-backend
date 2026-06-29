from __future__ import annotations

from app.models.song import SongPublic, SongRecord, to_public


def _make_record(**kwargs) -> SongRecord:
    defaults = {"job_id": "abc123", "title": "Test Song"}
    return SongRecord(**{**defaults, **kwargs})


class TestToPublic:
    def test_job_id_maps_to_id(self):
        record = _make_record(job_id="xyz789")
        public = to_public(record)
        assert public.id == "xyz789"

    def test_snake_case_fields_map_to_camelcase(self):
        record = _make_record(
            lrc_preview="[00:01.00] Hola",
            video_url="https://example.com/video.mp4",
            instrumental_url="https://example.com/inst.mp3",
        )
        public = to_public(record)
        assert public.lrcPreview == "[00:01.00] Hola"
        assert public.videoUrl == "https://example.com/video.mp4"
        assert public.instrumentalUrl == "https://example.com/inst.mp3"

    def test_optional_fields_are_none_when_not_set(self):
        record = _make_record()
        public = to_public(record)
        assert public.artist is None
        assert public.lrcPreview is None
        assert public.videoUrl is None
        assert public.instrumentalUrl is None
        assert public.lrc is None

    def test_model_dump_has_camelcase_keys(self):
        record = _make_record(lrc_preview="preview", video_url="http://v", instrumental_url="http://i")
        dumped = to_public(record).model_dump()
        assert "lrcPreview" in dumped
        assert "videoUrl" in dumped
        assert "instrumentalUrl" in dumped
        assert "lrc_preview" not in dumped
        assert "video_url" not in dumped


class TestSongRecord:
    def test_defaults(self):
        record = SongRecord(job_id="j1", title="T")
        assert record.bpm == 0
        assert record.status == "completed"
        assert record.tags == []

    def test_tags_default_is_independent(self):
        r1 = SongRecord(job_id="j1", title="T")
        r2 = SongRecord(job_id="j2", title="T")
        r1.tags.append("rock")
        assert r2.tags == []
