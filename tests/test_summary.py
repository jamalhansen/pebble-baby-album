"""Tests for summary.py — date bounds and summary generation orchestration."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest
from pebble import summary


def test_week_bounds():
    """Returns correct Monday-Sunday bounds."""
    # 2026-03-20 is Friday
    monday, sunday = summary._week_bounds(date(2026, 3, 20))
    assert monday == date(2026, 3, 16)
    assert sunday == date(2026, 3, 22)


def test_month_bounds():
    """Returns correct first-last day of month."""
    first, last = summary._month_bounds(date(2026, 3, 20))
    assert first == date(2026, 3, 1)
    assert last == date(2026, 3, 31)
    
    # Check December rollover
    first, last = summary._month_bounds(date(2026, 12, 10))
    assert first == date(2026, 12, 1)
    assert last == date(2026, 12, 31)


@patch("pebble.summary.iter_entries")
@patch("pebble.summary.asyncio.run")
def test_generate_summary_no_entries(mock_run, mock_iter, tmp_path):
    """Raises ValueError if no entries in range."""
    mock_iter.return_value = []
    with pytest.raises(ValueError, match="No entries found"):
        summary.generate_summary(tmp_path, MagicMock())


@patch("pebble.summary.iter_entries")
@patch("pebble.summary.asyncio.run")
@patch("pebble.summary.save_summary")
def test_generate_summary_success(mock_save, mock_run, mock_iter, tmp_path):
    """Orchestrates summary generation and saving."""
    mock_entry = MagicMock()
    mock_entry.date = date(2026, 3, 20)
    mock_iter.return_value = [mock_entry]
    
    mock_summary = MagicMock()
    mock_run.return_value = mock_summary
    
    res = summary.generate_summary(tmp_path, MagicMock(), ref_date=date(2026, 3, 20))
    
    assert res == mock_summary
    mock_save.assert_called_once_with(mock_summary, tmp_path)
    mock_run.assert_called_once()
