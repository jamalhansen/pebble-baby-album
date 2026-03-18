"""
One-off migration: convert stored HEIC files to JPEG and update journal references.

Finds all .heic / .HEIF files under the processed_dir, converts each to a
1024px-capped JPEG, deletes the original, then rewrites any journal markdown
files that reference the old HEIC path to point to the new .jpg path.

Usage:
    uv run scripts/migrate_heic_to_jpg.py
    uv run scripts/migrate_heic_to_jpg.py --dry-run
"""
import io
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from PIL import Image
from pillow_heif import register_heif_opener
from rich.console import Console

from pebble.config import load_config

register_heif_opener()

app = typer.Typer(help=__doc__)
console = Console()
err = Console(stderr=True, style="red")

_MAX_PX = 1024
_HEIC_SUFFIXES = {".heic", ".heif"}


def _convert(src: Path, dest: Path) -> None:
    """Open src (any PIL-supported format incl. HEIC), save as JPEG to dest."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((_MAX_PX, _MAX_PX), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    dest.write_bytes(buf.getvalue())


def _update_journal(journal_dir: Path, old_path: Path, new_path: Path, dry_run: bool) -> int:
    """Replace all occurrences of old_path string in markdown files with new_path."""
    old_str = str(old_path)
    new_str = str(new_path)
    updated = 0
    for md in journal_dir.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        if old_str in text:
            if dry_run:
                console.print(f"  [dim]Would update[/] {md.relative_to(journal_dir)}")
            else:
                md.write_text(text.replace(old_str, new_str), encoding="utf-8")
                console.print(f"  [dim]Updated[/] {md.relative_to(journal_dir)}")
            updated += 1
    return updated


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would change, don't write anything."),
    config_path: typer.FileText = typer.Option(None, "--config", "-C", help="Path to config.toml (overrides auto-detection)."),
) -> None:
    config = load_config(Path(config_path.name) if config_path else None)
    processed_dir = config.storage.processed_dir
    journal_dir = config.storage.journal_dir

    console.print(f"Scanning [cyan]{processed_dir}[/] for HEIC files…")

    heic_files = [
        p for p in processed_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _HEIC_SUFFIXES
    ]

    if not heic_files:
        console.print("[green]No HEIC files found — nothing to do.[/]")
        raise typer.Exit(0)

    console.print(f"Found [bold]{len(heic_files)}[/] HEIC file(s).\n")

    converted = 0
    skipped = 0

    for src in heic_files:
        dest = src.with_suffix(".jpg")
        rel = src.relative_to(processed_dir)
        console.print(f"[cyan]{rel}[/]")

        if dest.exists():
            console.print(f"  [yellow]⚠ Destination already exists, skipping:[/] {dest.name}")
            skipped += 1
            continue

        try:
            if dry_run:
                console.print(f"  [dim]Would convert → {dest.name}[/]")
                console.print(f"  [dim]Would delete  → {src.name}[/]")
            else:
                _convert(src, dest)
                src.unlink()
                console.print(f"  [green]✓[/] → {dest.name}")

            _update_journal(journal_dir, src, dest, dry_run)
            converted += 1

        except Exception as exc:
            err.print(f"  [red]✗ Failed:[/] {exc}")
            skipped += 1

    print()
    action = "Would convert" if dry_run else "Converted"
    console.print(f"Done. {action}: {converted}, Skipped: {skipped}")


if __name__ == "__main__":
    app()
