from __future__ import annotations

import pytest

from app.common.validators import ValidationError, require_file, require_title_artist


class TestRequireFile:
    def test_valid_input_passes(self):
        require_file(b"data", "song.mp3")

    def test_empty_bytes_raises(self):
        with pytest.raises(ValidationError, match="Missing file"):
            require_file(b"", "song.mp3")

    def test_none_bytes_raises(self):
        with pytest.raises(ValidationError, match="Missing file"):
            require_file(None, "song.mp3")

    def test_none_filename_raises(self):
        with pytest.raises(ValidationError, match="Missing file"):
            require_file(b"data", None)

    def test_empty_filename_raises(self):
        with pytest.raises(ValidationError, match="Missing file"):
            require_file(b"data", "")


class TestRequireTitleArtist:
    def test_valid_input_passes(self):
        require_title_artist("Mi canción", "El artista")

    def test_none_title_raises(self):
        with pytest.raises(ValidationError, match="Title and artist required"):
            require_title_artist(None, "Artista")

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="Title and artist required"):
            require_title_artist("", "Artista")

    def test_none_artist_raises(self):
        with pytest.raises(ValidationError, match="Title and artist required"):
            require_title_artist("Título", None)

    def test_empty_artist_raises(self):
        with pytest.raises(ValidationError, match="Title and artist required"):
            require_title_artist("Título", "")
