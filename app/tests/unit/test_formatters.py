from __future__ import annotations

from app.common.formatters import (
    build_preview,
    clamp_progress,
    format_duration,
    normalize_tags,
)


class TestFormatDuration:
    def test_none_returns_placeholder(self):
        assert format_duration(None) == "--:--"

    def test_negative_returns_placeholder(self):
        assert format_duration(-1) == "--:--"

    def test_zero(self):
        assert format_duration(0) == "0:00"

    def test_seconds_only(self):
        assert format_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert format_duration(65) == "1:05"

    def test_padding(self):
        assert format_duration(125) == "2:05"

    def test_hours(self):
        assert format_duration(3661) == "1:01:01"

    def test_rounding(self):
        assert format_duration(125.6) == "2:06"


class TestBuildPreview:
    def test_empty_string(self):
        assert build_preview("") == ""

    def test_whitespace_only(self):
        assert build_preview("   \n   ") == ""

    def test_first_line(self):
        assert build_preview("Línea uno\nLínea dos") == "Línea uno"

    def test_skips_empty_leading_lines(self):
        assert build_preview("\n  \nLínea tres") == "Línea tres"

    def test_lrc_line_preserved(self):
        assert build_preview("[00:12.34]Primera palabra") == "[00:12.34]Primera palabra"


class TestNormalizeTags:
    def test_empty_string_returns_default(self):
        assert normalize_tags("") == ["subido"]

    def test_whitespace_only_returns_default(self):
        assert normalize_tags("   ") == ["subido"]

    def test_single_tag(self):
        assert normalize_tags("rock") == ["rock"]

    def test_multiple_tags(self):
        assert normalize_tags("rock, pop") == ["rock", "pop"]

    def test_strips_whitespace(self):
        assert normalize_tags("  rock  ,  pop  ") == ["rock", "pop"]

    def test_ignores_empty_segments(self):
        assert normalize_tags("rock,  , pop, ") == ["rock", "pop"]


class TestClampProgress:
    def test_below_zero(self):
        assert clamp_progress(-10) == 0

    def test_zero(self):
        assert clamp_progress(0) == 0

    def test_midpoint(self):
        assert clamp_progress(50) == 50

    def test_hundred(self):
        assert clamp_progress(100) == 100

    def test_above_hundred(self):
        assert clamp_progress(150) == 100
