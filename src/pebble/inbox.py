"""Photo inbox — scan a folder, describe each photo, file it into the journal."""
import asyncio
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from PIL import Image
from rich.console import Console

from .agents import describe_photo
from .config import Config
from .models import JournalEntry, Mood
from .storage import append_entry

console = Console()
err_console = Console(stderr=True, style="red")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def get_photo_date(image_path: Path) -> date:
    """
    Return the date a photo was taken.

    Priority:
    1. EXIF DateTimeOriginal (tag 36867)
    2. File modification time
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()  # type: ignore[attr-defined]
            if exif:
                raw = exif.get(36867)  # DateTimeOriginal
                if raw:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").date()
    except Exception:
        pass

    # Fall back to file modification time
    return datetime.fromtimestamp(image_path.stat().st_mtime).date()


def iter_inbox(inbox_dir: Path) -> Iterator[Path]:
    """Yield image files in the inbox directory, sorted by filename."""
    if not inbox_dir.exists():
        return
    paths = [
        p for p in inbox_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    yield from sorted(paths, key=lambda p: p.name)


def _dest_path(processed_dir: Path, photo_date: date, original: Path) -> Path:
    """
    Build the destination path, handling filename collisions with a counter suffix.

    e.g. processed/2025-12-15/photo.jpg  →  processed/2025-12-15/photo-1.jpg
    """
    date_dir = processed_dir / photo_date.isoformat()
    date_dir.mkdir(parents=True, exist_ok=True)
    dest = date_dir / original.name
    if not dest.exists():
        return dest
    stem, suffix = original.stem, original.suffix
    counter = 1
    while True:
        candidate = date_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def process_inbox(
    config: Config,
    dry_run: bool = False,
    verbose: bool = False,
    model=None,
) -> tuple[int, int]:
    """
    Process all photos in the inbox directory.

    For each photo:
    - Determine date from EXIF (fallback: mtime)
    - Describe it with the vision agent
    - Append a journal entry for that date
    - Move photo to processed_dir/YYYY-MM-DD/

    Returns (processed_count, skipped_count).
    """
    inbox_dir = config.storage.inbox_dir
    photos = list(iter_inbox(inbox_dir))

    if not photos:
        console.print(f"[dim]Inbox is empty: {inbox_dir}[/]")
        return 0, 0

    total = len(photos)
    processed = 0
    skipped = 0

    for i, image_path in enumerate(photos, 1):
        prefix = f"[dim][{i}/{total}][/]"
        try:
            photo_date = get_photo_date(image_path)
            console.print(f"{prefix} [cyan]{photo_date}[/] — {image_path.name}")

            photo_desc = asyncio.run(describe_photo(image_path, config, model=model))

            if verbose:
                console.print(f"  [dim]{photo_desc.description[:120]}…[/]")

            age_weeks = config.age_weeks(photo_date)
            entry = JournalEntry(
                date=photo_date,
                age_weeks=age_weeks,
                milestone_tags=[],
                mood=Mood.TENDER,
                raw_input=f"[inbox: {image_path.name}]",
                narrative=f"Photo added from inbox: {image_path.name}",
                photos=[photo_desc],
            )

            if not dry_run:
                append_entry(entry, config.storage.journal_dir)
                dest = _dest_path(config.storage.processed_dir, photo_date, image_path)
                shutil.move(str(image_path), dest)
                console.print(f"  [green]✓[/] → {dest.relative_to(config.storage.processed_dir.parent)}")

            processed += 1

        except Exception as exc:
            err_console.print(f"  [red]✗ Skipped {image_path.name}:[/] {exc}")
            skipped += 1

    return processed, skipped
