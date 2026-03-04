"""Tests for inbox.py — photo inbox scanning and batch processing."""
import asyncio
import shutil
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pebble.config import Config, BabyConfig, ModelsConfig, StorageConfig, WebConfig
from pebble.models import JournalEntry, MilestoneTag, Mood, PhotoDescription
from pebble.inbox import get_photo_date, iter_inbox, process_inbox, _dest_path

BIRTH_DATE = date(2024, 1, 1)

# Minimal valid JPEG bytes (SOI + APP0 marker)
DUMMY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def make_config(tmp_path: Path) -> Config:
    inbox_dir = tmp_path / "inbox"
    processed_dir = tmp_path / "inbox" / "processed"
    return Config(
        baby=BabyConfig(name="Test Baby", birth_date=BIRTH_DATE),
        models=ModelsConfig(
            text_model="qwen2.5:3b",
            vision_model="llava:7b",
            ollama_host="http://localhost:11434",
        ),
        storage=StorageConfig(
            journal_dir=tmp_path / "journal",
            inbox_dir=inbox_dir,
            processed_dir=processed_dir,
        ),
        web=WebConfig(port=5555),
    )


def make_photo(directory: Path, name: str = "photo.jpg") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(DUMMY_JPEG)
    return path


class TestGetPhotoDate:
    def test_falls_back_to_mtime_when_no_exif(self, tmp_path):
        photo = make_photo(tmp_path)
        # No EXIF in dummy bytes — should fall back to mtime
        result = get_photo_date(photo)
        assert isinstance(result, date)

    def test_reads_exif_date_when_present(self, tmp_path):
        photo = make_photo(tmp_path)
        exif_date = date(2025, 12, 15)

        mock_exif = {36867: "2025:12:15 10:30:00"}
        mock_img = MagicMock()
        mock_img._getexif.return_value = mock_exif
        mock_img.__enter__ = lambda s: s
        mock_img.__exit__ = MagicMock(return_value=False)

        with patch("pebble.inbox.Image.open", return_value=mock_img):
            result = get_photo_date(photo)

        assert result == exif_date

    def test_falls_back_when_exif_missing_tag(self, tmp_path):
        photo = make_photo(tmp_path)

        mock_exif = {}  # No DateTimeOriginal tag
        mock_img = MagicMock()
        mock_img._getexif.return_value = mock_exif
        mock_img.__enter__ = lambda s: s
        mock_img.__exit__ = MagicMock(return_value=False)

        with patch("pebble.inbox.Image.open", return_value=mock_img):
            result = get_photo_date(photo)

        assert isinstance(result, date)

    def test_falls_back_when_exif_raises(self, tmp_path):
        photo = make_photo(tmp_path)

        with patch("pebble.inbox.Image.open", side_effect=Exception("corrupt")):
            result = get_photo_date(photo)

        assert isinstance(result, date)


class TestIterInbox:
    def test_yields_nothing_when_dir_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert list(iter_inbox(missing)) == []

    def test_yields_supported_image_types(self, tmp_path):
        inbox = tmp_path / "inbox"
        for name in ["a.jpg", "b.jpeg", "c.png", "d.gif", "e.webp"]:
            make_photo(inbox, name)

        results = list(iter_inbox(inbox))
        assert len(results) == 5

    def test_ignores_non_image_files(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "notes.txt").write_text("not a photo")
        (inbox / "data.json").write_text("{}")
        make_photo(inbox, "real.jpg")

        results = list(iter_inbox(inbox))
        assert len(results) == 1
        assert results[0].name == "real.jpg"

    def test_sorted_by_filename(self, tmp_path):
        inbox = tmp_path / "inbox"
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            make_photo(inbox, name)

        results = list(iter_inbox(inbox))
        assert [p.name for p in results] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_ignores_files_in_subdirectories(self, tmp_path):
        inbox = tmp_path / "inbox"
        make_photo(inbox, "top.jpg")
        make_photo(inbox / "processed", "nested.jpg")

        results = list(iter_inbox(inbox))
        assert len(results) == 1
        assert results[0].name == "top.jpg"


class TestDestPath:
    def test_simple_path(self, tmp_path):
        processed = tmp_path / "processed"
        photo_date = date(2025, 12, 15)
        original = Path("photo.jpg")

        dest = _dest_path(processed, photo_date, original)
        assert dest == processed / "2025-12-15" / "photo.jpg"
        assert (processed / "2025-12-15").exists()

    def test_handles_collision(self, tmp_path):
        processed = tmp_path / "processed"
        photo_date = date(2025, 12, 15)
        original = Path("photo.jpg")

        # Create the conflicting file
        date_dir = processed / "2025-12-15"
        date_dir.mkdir(parents=True)
        (date_dir / "photo.jpg").write_bytes(b"existing")

        dest = _dest_path(processed, photo_date, original)
        assert dest.name == "photo-1.jpg"

    def test_handles_multiple_collisions(self, tmp_path):
        processed = tmp_path / "processed"
        photo_date = date(2025, 12, 15)
        original = Path("photo.jpg")

        date_dir = processed / "2025-12-15"
        date_dir.mkdir(parents=True)
        (date_dir / "photo.jpg").write_bytes(b"existing")
        (date_dir / "photo-1.jpg").write_bytes(b"existing")

        dest = _dest_path(processed, photo_date, original)
        assert dest.name == "photo-2.jpg"


class TestProcessInbox:
    def _make_photo_desc(self, image_path: Path) -> PhotoDescription:
        return PhotoDescription(
            file_path=str(image_path),
            description="A cute baby smiling at the camera.",
        )

    def test_processes_photo_and_moves_it(self, tmp_path):
        config = make_config(tmp_path)
        photo = make_photo(config.storage.inbox_dir)

        fixed_desc = self._make_photo_desc(photo)

        with patch("pebble.inbox.describe_photo", new=AsyncMock(return_value=fixed_desc)), \
             patch("pebble.inbox.append_entry") as mock_append, \
             patch("pebble.inbox.get_photo_date", return_value=date(2025, 12, 15)):

            processed, skipped = process_inbox(config)

        assert processed == 1
        assert skipped == 0
        mock_append.assert_called_once()
        # Photo should be moved out of inbox
        assert not photo.exists()

    def test_dry_run_does_not_save_or_move(self, tmp_path):
        config = make_config(tmp_path)
        photo = make_photo(config.storage.inbox_dir)

        fixed_desc = self._make_photo_desc(photo)

        with patch("pebble.inbox.describe_photo", new=AsyncMock(return_value=fixed_desc)), \
             patch("pebble.inbox.append_entry") as mock_append, \
             patch("pebble.inbox.get_photo_date", return_value=date(2025, 12, 15)):

            processed, skipped = process_inbox(config, dry_run=True)

        assert processed == 1
        assert skipped == 0
        mock_append.assert_not_called()
        assert photo.exists()  # not moved

    def test_skips_on_error_and_continues(self, tmp_path):
        config = make_config(tmp_path)
        photo1 = make_photo(config.storage.inbox_dir, "a.jpg")
        photo2 = make_photo(config.storage.inbox_dir, "b.jpg")

        call_count = 0

        async def flaky_describe(path, config, model=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Ollama error")
            return PhotoDescription(file_path=str(path), description="Nice photo.")

        with patch("pebble.inbox.describe_photo", new=flaky_describe), \
             patch("pebble.inbox.append_entry"), \
             patch("pebble.inbox.get_photo_date", return_value=date(2025, 12, 15)):

            processed, skipped = process_inbox(config)

        assert processed == 1
        assert skipped == 1

    def test_returns_zero_zero_when_inbox_empty(self, tmp_path):
        config = make_config(tmp_path)
        config.storage.inbox_dir.mkdir(parents=True, exist_ok=True)

        processed, skipped = process_inbox(config)

        assert processed == 0
        assert skipped == 0

    def test_journal_entry_uses_exif_date(self, tmp_path):
        config = make_config(tmp_path)
        photo = make_photo(config.storage.inbox_dir)
        exif_date = date(2025, 11, 20)

        fixed_desc = self._make_photo_desc(photo)
        captured_entries = []

        def capture_append(entry, journal_dir):
            captured_entries.append(entry)
            return journal_dir / f"{entry.date}.md"

        with patch("pebble.inbox.describe_photo", new=AsyncMock(return_value=fixed_desc)), \
             patch("pebble.inbox.append_entry", side_effect=capture_append), \
             patch("pebble.inbox.get_photo_date", return_value=exif_date):

            process_inbox(config)

        assert len(captured_entries) == 1
        assert captured_entries[0].date == exif_date
        assert captured_entries[0].photos[0].description == fixed_desc.description

    def test_photo_moved_to_date_subdir(self, tmp_path):
        config = make_config(tmp_path)
        photo = make_photo(config.storage.inbox_dir)
        photo_date = date(2025, 12, 15)

        fixed_desc = self._make_photo_desc(photo)

        with patch("pebble.inbox.describe_photo", new=AsyncMock(return_value=fixed_desc)), \
             patch("pebble.inbox.append_entry"), \
             patch("pebble.inbox.get_photo_date", return_value=photo_date):

            process_inbox(config)

        expected = config.storage.processed_dir / "2025-12-15" / "photo.jpg"
        assert expected.exists()
