"""Weekly and monthly summary generation."""
import asyncio
from datetime import date, timedelta
from pathlib import Path

from .agents import summarize_entries
from .config import Config
from .models import WeeklySummary
from .storage import iter_entries, save_summary


def _week_bounds(ref: date) -> tuple[date, date]:
    """Return the Monday–Sunday bounds for the week containing ref."""
    start = ref - timedelta(days=ref.weekday())
    end = start + timedelta(days=6)
    return start, end


def _month_bounds(ref: date) -> tuple[date, date]:
    """Return the first–last day of the month containing ref."""
    start = ref.replace(day=1)
    if ref.month == 12:
        end = ref.replace(year=ref.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = ref.replace(month=ref.month + 1, day=1) - timedelta(days=1)
    return start, end


def generate_summary(
    journal_dir: Path,
    config: Config,
    week: bool = True,
    ref_date: date | None = None,
    write: bool = True,
    model=None,
) -> WeeklySummary:
    """Generate a weekly or monthly summary. Runs the async agent synchronously."""
    today = ref_date or date.today()
    if week:
        start, end = _week_bounds(today)
    else:
        start, end = _month_bounds(today)

    entries = [
        e for e in iter_entries(journal_dir)
        if start <= e.date <= end
    ]

    if not entries:
        raise ValueError(f"No entries found between {start.isoformat()} and {end.isoformat()}")

    summary = asyncio.run(
        summarize_entries(entries, start, end, config, model=model)
    )

    if write:
        save_summary(summary, journal_dir)

    return summary
