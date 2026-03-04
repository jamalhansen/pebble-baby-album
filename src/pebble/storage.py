"""Read and write journal entries as markdown files with YAML frontmatter."""
from datetime import date
from pathlib import Path
from typing import Iterator

import frontmatter

from .models import JournalEntry, MilestoneTag, Mood, PhotoDescription, WeeklySummary


def _entry_path(journal_dir: Path, entry_date: date) -> Path:
    return journal_dir / f"{entry_date.isoformat()}.md"


def _serialize_entry(entry: JournalEntry) -> str:
    """Convert a JournalEntry to markdown with YAML frontmatter."""
    post = frontmatter.Post(
        content=entry.narrative,
        date=entry.date.isoformat(),
        age_weeks=entry.age_weeks,
        milestone_tags=[tag.value for tag in entry.milestone_tags],
        mood=entry.mood.value,
        raw_input=entry.raw_input,
    )
    body = frontmatter.dumps(post)

    if entry.photos:
        photo_lines = ["\n\n## Photos"]
        for photo in entry.photos:
            photo_lines.append(f"\n### {photo.file_path}\n")
            photo_lines.append(photo.description)
        body += "\n".join(photo_lines)

    return body


def _parse_photos(content: str) -> tuple[str, list[PhotoDescription]]:
    """Split narrative from Photos section and parse photo descriptions."""
    photos: list[PhotoDescription] = []
    if "## Photos" not in content:
        return content.strip(), photos

    parts = content.split("## Photos", 1)
    narrative = parts[0].strip()
    photo_block = parts[1].strip()

    # Each photo starts with "### <path>"
    import re
    photo_sections = re.split(r"^### (.+)$", photo_block, flags=re.MULTILINE)
    # photo_sections: ["", path1, desc1, path2, desc2, ...]
    it = iter(photo_sections[1:])
    for path, desc in zip(it, it):
        photos.append(PhotoDescription(file_path=path.strip(), description=desc.strip()))

    return narrative, photos


def _parse_entry(path: Path) -> JournalEntry | None:
    """Parse a single markdown file into a JournalEntry. Returns None on error."""
    try:
        post = frontmatter.load(str(path))
    except Exception:
        return None

    meta = post.metadata
    entry_date = date.fromisoformat(str(meta.get("date", path.stem)))
    age_weeks = int(meta.get("age_weeks", 0))

    raw_tags = meta.get("milestone_tags", [])
    milestone_tags = []
    for t in raw_tags:
        try:
            milestone_tags.append(MilestoneTag(t))
        except ValueError:
            pass

    mood_raw = meta.get("mood", "tender")
    try:
        mood = Mood(mood_raw)
    except ValueError:
        mood = Mood.TENDER

    raw_input = str(meta.get("raw_input", ""))
    narrative, photos = _parse_photos(post.content)

    return JournalEntry(
        date=entry_date,
        age_weeks=age_weeks,
        milestone_tags=milestone_tags,
        mood=mood,
        raw_input=raw_input,
        narrative=narrative,
        photos=photos,
    )


def save_entry(entry: JournalEntry, journal_dir: Path) -> Path:
    """Write a JournalEntry to disk. Creates or overwrites the day's file."""
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(journal_dir, entry.date)
    path.write_text(_serialize_entry(entry), encoding="utf-8")
    return path


def append_entry(entry: JournalEntry, journal_dir: Path) -> Path:
    """
    Append a JournalEntry to an existing day's file, merging frontmatter.

    Milestone tags are unioned. Mood is updated to the new entry's mood.
    A --- separator is inserted between sections.
    """
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(journal_dir, entry.date)

    if not path.exists():
        return save_entry(entry, journal_dir)

    existing = _parse_entry(path)
    if existing is None:
        return save_entry(entry, journal_dir)

    # Merge tags (union) and update mood
    merged_tags = list({*existing.milestone_tags, *entry.milestone_tags})
    merged_mood = entry.mood

    # Rewrite frontmatter with merged values, keep existing narrative, append new
    post = frontmatter.load(str(path))
    post["milestone_tags"] = [t.value for t in merged_tags]
    post["mood"] = merged_mood.value

    existing_body = frontmatter.dumps(post)

    # Build new section body (no frontmatter, just narrative + photos)
    new_section = entry.narrative
    if entry.photos:
        import re as _re
        lines = ["\n\n## Photos"]
        for photo in entry.photos:
            lines.append(f"\n### {photo.file_path}\n")
            lines.append(photo.description)
        new_section += "\n".join(lines)

    full_content = existing_body.rstrip() + "\n\n---\n\n" + new_section + "\n"
    path.write_text(full_content, encoding="utf-8")
    return path


def load_entry(entry_date: date, journal_dir: Path) -> JournalEntry | None:
    """Load a single day's entry. Returns None if not found."""
    path = _entry_path(journal_dir, entry_date)
    if not path.exists():
        return None
    return _parse_entry(path)


def iter_entries(journal_dir: Path) -> Iterator[JournalEntry]:
    """Yield all journal entries, newest first."""
    paths = sorted(journal_dir.glob("????-??-??.md"), reverse=True)
    for path in paths:
        entry = _parse_entry(path)
        if entry is not None:
            yield entry


def search_entries(
    journal_dir: Path,
    query: str | None = None,
    tag: MilestoneTag | None = None,
    after: date | None = None,
    before: date | None = None,
) -> list[JournalEntry]:
    """Search entries by text and/or tag, optionally filtered by date range."""
    results = []
    for entry in iter_entries(journal_dir):
        if after and entry.date < after:
            continue
        if before and entry.date > before:
            continue
        if tag and tag not in entry.milestone_tags:
            continue
        if query:
            needle = query.lower()
            haystack = (entry.narrative + " " + entry.raw_input).lower()
            if needle not in haystack:
                continue
        results.append(entry)
    return results


def save_summary(summary: WeeklySummary, journal_dir: Path) -> Path:
    """Write a WeeklySummary to journal/summaries/."""
    summaries_dir = journal_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{summary.week_start.isoformat()}--{summary.week_end.isoformat()}.md"
    path = summaries_dir / filename

    lines = [
        f"# Week of {summary.week_start.isoformat()} – {summary.week_end.isoformat()}",
        "",
        summary.narrative,
        "",
        "## Highlights",
        "",
        *[f"- {h}" for h in summary.highlights],
        "",
        "## Milestones Reached",
        "",
        *[f"- {m}" for m in summary.milestones_reached],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
