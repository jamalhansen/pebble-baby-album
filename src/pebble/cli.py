"""Typer CLI for pebble baby journal."""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.markdown import Markdown

from .config import load_config
from .models import MilestoneTag, Mood, MOOD_EMOJI
from .storage import (
    append_entry,
    iter_entries,
    load_entry,
    search_entries,
)

app = typer.Typer(
    name="pebble",
    help="Privacy-first local baby journal. Everything stays on your machine.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True, style="red")


def _get_config(config_path: Optional[Path] = None):
    try:
        return load_config(config_path)
    except FileNotFoundError as e:
        err_console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def _run_ollama_check(config):
    """Warn the user if Ollama is not reachable."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(config.models.ollama_host, timeout=2)
    except Exception:
        err_console.print(
            f"[bold yellow]Warning:[/] Cannot reach Ollama at {config.models.ollama_host}\n"
            "Make sure Ollama is running: [bold]ollama serve[/]"
        )
        raise typer.Exit(1)


@app.command()
def log(
    note: Optional[str] = typer.Argument(None, help="Quick note (or omit to open $EDITOR)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
    date_str: Optional[str] = typer.Option(None, "--date", "-d", help="Entry date (YYYY-MM-DD)"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Override Ollama model (e.g. qwen2.5:7b)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print result without saving"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full structured output"),
):
    """Log a new journal entry from text."""
    config = _get_config(config_path)
    _run_ollama_check(config)

    entry_date = date.fromisoformat(date_str) if date_str else date.today()

    # Get text from argument, stdin, or editor
    if note:
        raw_text = note
    elif not sys.stdin.isatty():
        raw_text = sys.stdin.read().strip()
    else:
        import subprocess
        import tempfile
        import os
        editor = os.environ.get("EDITOR", "nano")
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            tmp_path = f.name
        subprocess.call([editor, tmp_path])
        raw_text = Path(tmp_path).read_text().strip()
        Path(tmp_path).unlink(missing_ok=True)

    if not raw_text:
        err_console.print("[red]No input provided.[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Processing entry for {entry_date.isoformat()}...[/]")

    from .agents import log_entry, model_from_name
    model = model_from_name(model_name, config) if model_name else None
    entry = asyncio.run(log_entry(raw_text, entry_date, config, model=model))

    if verbose:
        console.print(entry.model_dump_json(indent=2))

    if dry_run:
        console.print(Panel(entry.narrative, title=f"[bold]{entry_date}[/] (dry run)", border_style="yellow"))
        console.print(f"  Mood: {MOOD_EMOJI[entry.mood]} {entry.mood.value}")
        console.print(f"  Tags: {', '.join(t.value for t in entry.milestone_tags)}")
        return

    path = append_entry(entry, config.storage.journal_dir)
    console.print(f"[green]✓[/] Saved to {path}")
    console.print(f"  Mood: {MOOD_EMOJI[entry.mood]} {entry.mood.value}")
    console.print(f"  Tags: {', '.join(t.value for t in entry.milestone_tags)}")


@app.command()
def photo(
    image_path: Path = typer.Argument(..., help="Path to the photo"),
    note: Optional[str] = typer.Option(None, "--note", help="Optional text note to merge"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
    date_str: Optional[str] = typer.Option(None, "--date", "-d", help="Entry date (YYYY-MM-DD)"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Override Ollama vision model (e.g. llava:13b)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print result without saving"),
):
    """Add a photo (and optional note) to today's entry."""
    config = _get_config(config_path)
    _run_ollama_check(config)

    if not image_path.exists():
        err_console.print(f"[red]Image not found:[/] {image_path}")
        raise typer.Exit(1)

    entry_date = date.fromisoformat(date_str) if date_str else date.today()

    console.print(f"[dim]Describing photo {image_path.name}...[/]")

    from .agents import describe_photo, log_entry, model_from_name
    model = model_from_name(model_name, config) if model_name else None
    photo_desc = asyncio.run(describe_photo(image_path, config, model=model))

    console.print(f"[dim]Photo described.[/]")

    if note:
        console.print(f"[dim]Processing text note...[/]")
        entry = asyncio.run(log_entry(note, entry_date, config, model=model))
        entry.photos.append(photo_desc)
    else:
        # Create a minimal entry that just records the photo
        from .models import JournalEntry, Mood
        age_weeks = config.age_weeks(entry_date)
        entry = JournalEntry(
            date=entry_date,
            age_weeks=age_weeks,
            milestone_tags=[],
            mood=Mood.TENDER,
            raw_input=f"[photo: {image_path}]",
            narrative=f"Photo added: {image_path.name}",
            photos=[photo_desc],
        )

    if dry_run:
        console.print(Panel(photo_desc.description, title=f"Photo: {image_path.name} (dry run)", border_style="yellow"))
        return

    path = append_entry(entry, config.storage.journal_dir)
    console.print(f"[green]✓[/] Saved to {path}")


@app.command()
def recent(
    weeks: int = typer.Option(1, "--weeks", "-w", help="Number of weeks to show"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
):
    """Show a compact timeline of recent entries."""
    config = _get_config(config_path)
    cutoff = date.today() - timedelta(weeks=weeks)

    entries = [e for e in iter_entries(config.storage.journal_dir) if e.date >= cutoff]

    if not entries:
        console.print("[dim]No entries in this period.[/]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Wk", style="dim", justify="right")
    table.add_column("Mood", no_wrap=True)
    table.add_column("Tags", style="dim")
    table.add_column("Preview")

    for entry in entries:
        preview = entry.narrative.split("\n")[0][:60]
        if len(entry.narrative) > 60:
            preview += "…"
        table.add_row(
            entry.date.isoformat(),
            str(entry.age_weeks),
            f"{MOOD_EMOJI[entry.mood]} {entry.mood.value}",
            ", ".join(t.value for t in entry.milestone_tags),
            preview,
        )

    console.print(table)


@app.command()
def search(
    query: Optional[str] = typer.Argument(None, help="Full-text search query"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by milestone tag"),
    after: Optional[str] = typer.Option(None, "--after", "-a", help="Only entries after date (YYYY-MM-DD)"),
    before: Optional[str] = typer.Option(None, "--before", "-b", help="Only entries before date (YYYY-MM-DD)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
):
    """Search journal entries by text and/or tag."""
    config = _get_config(config_path)

    tag_enum = None
    if tag:
        try:
            tag_enum = MilestoneTag(tag)
        except ValueError:
            valid = ", ".join(t.value for t in MilestoneTag)
            err_console.print(f"[red]Unknown tag:[/] {tag}\nValid tags: {valid}")
            raise typer.Exit(1)

    after_date = date.fromisoformat(after) if after else None
    before_date = date.fromisoformat(before) if before else None

    results = search_entries(
        config.storage.journal_dir,
        query=query,
        tag=tag_enum,
        after=after_date,
        before=before_date,
    )

    if not results:
        console.print("[dim]No results found.[/]")
        return

    for entry in results:
        emoji = MOOD_EMOJI[entry.mood]
        tags_str = ", ".join(t.value for t in entry.milestone_tags)
        console.print(f"[bold cyan]{entry.date.isoformat()}[/]  {emoji}  [dim]{tags_str}[/]")
        console.print(f"  {entry.narrative.split(chr(10))[0][:80]}")
        console.print()

    console.print(f"[dim]{len(results)} result(s)[/]")


@app.command()
def view(
    entry_date_str: Optional[str] = typer.Argument(None, help="Date to view (YYYY-MM-DD), defaults to today"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
):
    """Pretty-print a single day's journal entry."""
    config = _get_config(config_path)
    entry_date = date.fromisoformat(entry_date_str) if entry_date_str else date.today()
    entry = load_entry(entry_date, config.storage.journal_dir)

    if entry is None:
        console.print(f"[dim]No entry for {entry_date.isoformat()}[/]")
        raise typer.Exit(0)

    emoji = MOOD_EMOJI[entry.mood]
    tags_str = "  ".join(f"[bold]{t.value}[/]" for t in entry.milestone_tags)
    header = f"{emoji} {entry.date.isoformat()}  ·  week {entry.age_weeks}  ·  {entry.mood.value}"
    console.print(Panel(header, style="cyan"))
    if tags_str:
        console.print(f"  {tags_str}\n")

    console.print(Markdown(entry.narrative))

    for photo in entry.photos:
        console.print()
        console.print(Panel(
            photo.description,
            title=f"[dim]📷 {photo.file_path}[/]",
            border_style="dim",
        ))


@app.command()
def summary(
    week: bool = typer.Option(False, "--week", "-w", help="Summarize current week"),
    month: bool = typer.Option(False, "--month", help="Summarize current month"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Override Ollama model (e.g. qwen2.5:7b)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print result without saving to disk"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
):
    """Generate a weekly or monthly summary."""
    if not week and not month:
        err_console.print("[red]Specify --week or --month[/]")
        raise typer.Exit(1)

    config = _get_config(config_path)
    _run_ollama_check(config)

    from .agents import model_from_name
    from .summary import generate_summary
    model = model_from_name(model_name, config) if model_name else None
    try:
        result = generate_summary(
            config.storage.journal_dir,
            config,
            week=week,
            write=not dry_run,
            model=model,
        )
    except ValueError as e:
        err_console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    console.print(Panel(result.narrative, title="Summary", border_style="cyan"))
    console.print("\n[bold]Highlights[/]")
    for h in result.highlights:
        console.print(f"  • {h}")
    console.print("\n[bold]Milestones[/]")
    for m in result.milestones_reached:
        console.print(f"  🌟 {m}")


@app.command()
def inbox(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Override Ollama vision model (e.g. moondream)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be processed without making changes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print photo descriptions as they're generated"),
):
    """Process all photos in the inbox folder and file them into the journal."""
    config = _get_config(config_path)
    _run_ollama_check(config)

    inbox_dir = config.storage.inbox_dir
    processed_dir = config.storage.processed_dir

    console.print(f"[dim]Inbox:     {inbox_dir}[/]")
    if dry_run:
        console.print("[yellow]Dry run — no files will be moved or saved.[/]")
    else:
        console.print(f"[dim]Processed: {processed_dir}[/]")

    from .agents import model_from_name
    from .inbox import process_inbox
    model = model_from_name(model_name, config) if model_name else None
    processed, skipped = process_inbox(config, dry_run=dry_run, verbose=verbose, model=model)

    suffix = " (dry run)" if dry_run else ""
    console.print(f"\n[bold]Done{suffix}.[/] Processed: {processed}, Skipped: {skipped}")


@app.command()
def serve(
    port: int = typer.Option(5555, "--port", "-p", help="Port to listen on"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.toml"),
):
    """Start the local web viewer."""
    config = _get_config(config_path)
    actual_port = port or config.web.port

    console.print(f"[bold green]pebble web viewer[/] starting at [cyan]http://localhost:{actual_port}[/]")
    console.print("[dim]Press Ctrl+C to stop.[/]")

    from web.app import create_app
    flask_app = create_app(config)
    flask_app.run(host="127.0.0.1", port=actual_port, debug=False)


if __name__ == "__main__":
    app()
