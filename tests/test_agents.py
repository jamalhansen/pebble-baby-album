"""Tests for agents.py — uses MockProvider to avoid real Ollama calls."""
import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import patch

from pebble.models import EntryMetadata, JournalEntry, MilestoneTag, Mood, PhotoDescription, WeeklySummary
from pebble.config import Config, BabyConfig, ModelsConfig, StorageConfig, WebConfig
from local_first_common.testing import MockProvider

BIRTH_DATE = date(2024, 1, 1)


def make_config(tmp_path: Path) -> Config:
    return Config(
        baby=BabyConfig(name="Test Baby", birth_date=BIRTH_DATE),
        models=ModelsConfig(
            text_model="qwen2.5:3b",
            vision_model="llava:7b",
            ollama_host="http://localhost:11434",
        ),
        storage=StorageConfig(
            journal_dir=tmp_path,
            inbox_dir=tmp_path / "inbox",
            processed_dir=tmp_path / "inbox" / "processed",
        ),
        web=WebConfig(port=5555),
    )


def make_journal_entry(entry_date: date, config: Config) -> JournalEntry:
    return JournalEntry(
        date=entry_date,
        age_weeks=config.age_weeks(entry_date),
        milestone_tags=[MilestoneTag.MOTOR_SKILLS, MilestoneTag.FIRST],
        mood=Mood.PROUD,
        raw_input="grabbed my finger",
        narrative="He grabbed my finger tightly during tummy time.",
    )


class TestLogEntry:
    def _meta_response(self):
        return EntryMetadata(
            milestone_tags=[MilestoneTag.MOTOR_SKILLS, MilestoneTag.FIRST],
            mood=Mood.PROUD,
        ).model_dump_json()

    def test_log_entry_sets_date_and_age(self, tmp_path):
        config = make_config(tmp_path)
        entry_date = date(2025, 12, 15)
        expected_age = config.age_weeks(entry_date)

        mock_llm = MockProvider(response=self._meta_response())
        with patch("pebble.agents._get_provider", return_value=mock_llm):
            from pebble import agents
            entry = asyncio.run(agents.log_entry("grabbed my finger", entry_date, config))

        assert entry.date == entry_date
        assert entry.age_weeks == expected_age
        assert entry.raw_input == "grabbed my finger"

    def test_log_entry_uses_raw_text_as_narrative(self, tmp_path):
        config = make_config(tmp_path)
        entry_date = date(2025, 12, 15)

        mock_llm = MockProvider(response=self._meta_response())
        with patch("pebble.agents._get_provider", return_value=mock_llm):
            from pebble import agents
            entry = asyncio.run(agents.log_entry("grabbed my finger", entry_date, config))

        assert entry.narrative == "grabbed my finger"

    def test_log_entry_preserves_tags(self, tmp_path):
        config = make_config(tmp_path)
        entry_date = date(2025, 12, 15)

        mock_llm = MockProvider(response=self._meta_response())
        with patch("pebble.agents._get_provider", return_value=mock_llm):
            from pebble import agents
            entry = asyncio.run(agents.log_entry("grabbed my finger", entry_date, config))

        assert MilestoneTag.MOTOR_SKILLS in entry.milestone_tags
        assert MilestoneTag.FIRST in entry.milestone_tags

    def test_log_entry_passes_baby_context_in_prompt(self, tmp_path):
        config = make_config(tmp_path)
        entry_date = date(2025, 12, 15)

        mock_llm = MockProvider(response=self._meta_response())
        with patch("pebble.agents._get_provider", return_value=mock_llm):
            from pebble import agents
            asyncio.run(agents.log_entry("grabbed my finger", entry_date, config))

        system, user = mock_llm.calls[0]
        assert "Test Baby" in user
        assert "2025-12-15" in user


class TestDescribePhoto:
    def test_describe_photo_sets_file_path(self, tmp_path):
        config = make_config(tmp_path)

        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        from pebble.models import PhotoAnalysis
        fixed_analysis = PhotoAnalysis(
            description="Baby lying on a white blanket, smiling at camera.",
        )

        dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 16

        mock_llm = MockProvider(response=fixed_analysis.model_dump_json())
        with patch("pebble.agents._get_provider", return_value=mock_llm), \
             patch("pebble.agents._to_jpeg_bytes", return_value=(dummy_jpeg, (3024, 4032), (768, 1024))):
            from pebble import agents
            photo = asyncio.run(agents.describe_photo(image_path, config))

        assert photo.file_path == str(image_path)
        assert "smiling" in photo.description


class TestSummarizeEntries:
    def test_summarize_sets_date_range(self, tmp_path):
        config = make_config(tmp_path)
        week_start = date(2025, 12, 15)
        week_end = date(2025, 12, 21)

        entries = [make_journal_entry(date(2025, 12, 15 + i), config) for i in range(3)]

        fixed_summary = WeeklySummary(
            week_start=date(2025, 1, 1),
            week_end=date(2025, 1, 7),
            highlights=["Great week"],
            milestones_reached=["First grip"],
            narrative="A wonderful week.",
        )

        mock_llm = MockProvider(response=fixed_summary.model_dump_json())
        with patch("pebble.agents._get_provider", return_value=mock_llm):
            from pebble import agents
            summary = asyncio.run(
                agents.summarize_entries(entries, week_start, week_end, config)
            )

        assert summary.week_start == week_start
        assert summary.week_end == week_end


class TestConfig:
    def test_age_weeks_zero_at_birth(self):
        config = Config(
            baby=BabyConfig(name="Test", birth_date=date(2025, 11, 15)),
            models=ModelsConfig("m", "v", "http://localhost:11434"),
            storage=StorageConfig(Path("/tmp"), Path("/tmp/inbox"), Path("/tmp/inbox/processed")),
            web=WebConfig(5555),
        )
        assert config.age_weeks(date(2025, 11, 15)) == 0

    def test_age_weeks_four_weeks(self):
        config = Config(
            baby=BabyConfig(name="Test", birth_date=date(2025, 11, 15)),
            models=ModelsConfig("m", "v", "http://localhost:11434"),
            storage=StorageConfig(Path("/tmp"), Path("/tmp/inbox"), Path("/tmp/inbox/processed")),
            web=WebConfig(5555),
        )
        assert config.age_weeks(date(2025, 12, 13)) == 4

    def test_age_weeks_never_negative(self):
        config = Config(
            baby=BabyConfig(name="Test", birth_date=date(2025, 11, 15)),
            models=ModelsConfig("m", "v", "http://localhost:11434"),
            storage=StorageConfig(Path("/tmp"), Path("/tmp/inbox"), Path("/tmp/inbox/processed")),
            web=WebConfig(5555),
        )
        assert config.age_weeks(date(2025, 11, 1)) == 0
