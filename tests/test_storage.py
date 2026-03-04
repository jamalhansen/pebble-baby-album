"""Tests for storage.py — reading, writing, searching journal entries."""
from datetime import date
from pathlib import Path

import pytest

from pebble.models import JournalEntry, MilestoneTag, Mood, PhotoDescription
from pebble.storage import (
    append_entry,
    iter_entries,
    load_entry,
    save_entry,
    search_entries,
    save_summary,
)
from pebble.models import WeeklySummary

FIXTURES = Path(__file__).parent / "fixtures"


class TestSaveAndLoad:
    def test_save_creates_file(self, tmp_path):
        entry = JournalEntry(
            date=date(2025, 12, 15),
            age_weeks=4,
            milestone_tags=[MilestoneTag.MOTOR_SKILLS],
            mood=Mood.PROUD,
            raw_input="grabbed my finger",
            narrative="He grabbed my finger.",
        )
        path = save_entry(entry, tmp_path)
        assert path.exists()
        assert path.name == "2025-12-15.md"

    def test_load_roundtrip(self, tmp_path):
        entry = JournalEntry(
            date=date(2025, 12, 15),
            age_weeks=4,
            milestone_tags=[MilestoneTag.MOTOR_SKILLS, MilestoneTag.FIRST],
            mood=Mood.PROUD,
            raw_input="grabbed my finger",
            narrative="He grabbed my finger and wouldn't let go.",
        )
        save_entry(entry, tmp_path)
        loaded = load_entry(date(2025, 12, 15), tmp_path)
        assert loaded is not None
        assert loaded.date == entry.date
        assert loaded.age_weeks == entry.age_weeks
        assert set(loaded.milestone_tags) == set(entry.milestone_tags)
        assert loaded.mood == entry.mood
        assert "grabbed my finger" in loaded.narrative

    def test_load_missing_returns_none(self, tmp_path):
        result = load_entry(date(2025, 1, 1), tmp_path)
        assert result is None

    def test_load_fixture(self):
        entry = load_entry(date(2025, 12, 15), FIXTURES)
        assert entry is not None
        assert entry.mood == Mood.PROUD
        assert MilestoneTag.FIRST in entry.milestone_tags
        assert MilestoneTag.MOTOR_SKILLS in entry.milestone_tags
        assert len(entry.photos) == 1
        assert "~/photos/baby-grip.jpg" in entry.photos[0].file_path

    def test_load_fixture_with_photos(self):
        entry = load_entry(date(2025, 12, 15), FIXTURES)
        assert entry is not None
        assert entry.photos[0].description.startswith("Baby in a white onesie")


class TestAppend:
    def test_append_creates_new_file_if_not_exists(self, tmp_path):
        entry = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[MilestoneTag.SLEEP],
            mood=Mood.TIRED,
            raw_input="rough night",
            narrative="Rough night.",
        )
        path = append_entry(entry, tmp_path)
        assert path.exists()

    def test_append_merges_tags(self, tmp_path):
        entry1 = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[MilestoneTag.SLEEP],
            mood=Mood.TIRED,
            raw_input="rough night",
            narrative="Rough night.",
        )
        entry2 = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[MilestoneTag.FEEDING],
            mood=Mood.GRATEFUL,
            raw_input="good feed",
            narrative="Great morning feed.",
        )
        append_entry(entry1, tmp_path)
        append_entry(entry2, tmp_path)

        loaded = load_entry(date(2025, 12, 16), tmp_path)
        assert loaded is not None
        tags = set(loaded.milestone_tags)
        assert MilestoneTag.SLEEP in tags
        assert MilestoneTag.FEEDING in tags

    def test_append_updates_mood(self, tmp_path):
        entry1 = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[],
            mood=Mood.TIRED,
            raw_input="tired",
            narrative="Very tired.",
        )
        entry2 = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[],
            mood=Mood.JOYFUL,
            raw_input="better now",
            narrative="Feeling better.",
        )
        append_entry(entry1, tmp_path)
        append_entry(entry2, tmp_path)

        loaded = load_entry(date(2025, 12, 16), tmp_path)
        assert loaded is not None
        assert loaded.mood == Mood.JOYFUL

    def test_append_adds_separator(self, tmp_path):
        entry1 = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[],
            mood=Mood.TIRED,
            raw_input="morning",
            narrative="Morning entry.",
        )
        entry2 = JournalEntry(
            date=date(2025, 12, 16),
            age_weeks=4,
            milestone_tags=[],
            mood=Mood.JOYFUL,
            raw_input="evening",
            narrative="Evening entry.",
        )
        append_entry(entry1, tmp_path)
        append_entry(entry2, tmp_path)

        content = (tmp_path / "2025-12-16.md").read_text()
        assert "---" in content


class TestSearch:
    def test_search_by_text(self):
        results = search_entries(FIXTURES, query="grip")
        assert len(results) >= 1
        assert any("grip" in e.narrative.lower() or "grip" in e.raw_input.lower() for e in results)

    def test_search_by_tag(self):
        results = search_entries(FIXTURES, tag=MilestoneTag.FIRST)
        assert len(results) >= 2
        assert all(MilestoneTag.FIRST in e.milestone_tags for e in results)

    def test_search_after_date(self):
        results = search_entries(FIXTURES, after=date(2026, 1, 1))
        assert all(e.date >= date(2026, 1, 1) for e in results)

    def test_search_no_results(self):
        results = search_entries(FIXTURES, query="xyzzy_not_in_any_entry")
        assert results == []

    def test_search_combined(self):
        results = search_entries(FIXTURES, tag=MilestoneTag.FIRST, after=date(2025, 12, 18))
        assert all(MilestoneTag.FIRST in e.milestone_tags for e in results)
        assert all(e.date >= date(2025, 12, 18) for e in results)


class TestIterEntries:
    def test_newest_first(self):
        entries = list(iter_entries(FIXTURES))
        dates = [e.date for e in entries]
        assert dates == sorted(dates, reverse=True)

    def test_counts_fixtures(self):
        entries = list(iter_entries(FIXTURES))
        assert len(entries) == 5


class TestSaveSummary:
    def test_saves_to_summaries_dir(self, tmp_path):
        summary = WeeklySummary(
            week_start=date(2025, 12, 15),
            week_end=date(2025, 12, 21),
            highlights=["First real smile", "Bottle feeding going well"],
            milestones_reached=["First laugh"],
            narrative="A wonderful week full of firsts.",
        )
        path = save_summary(summary, tmp_path)
        assert path.exists()
        assert "summaries" in str(path)
        content = path.read_text()
        assert "First laugh" in content
