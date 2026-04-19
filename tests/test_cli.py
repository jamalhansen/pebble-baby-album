"""Tests for cli.py — command line interface for pebble."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest
import typer
from pebble import cli
from pebble.models import Mood
from pebble.cli import PebbleError, OllamaUnreachableError, StorageError


class TestTypedErrors:
    def test_pebble_error_hierarchy(self):
        err = PebbleError("base")
        assert isinstance(err, Exception)

    def test_ollama_unreachable_is_pebble_error(self):
        err = OllamaUnreachableError("ollama down")
        assert isinstance(err, PebbleError)
        assert "ollama down" in str(err)

    def test_storage_error_is_pebble_error(self):
        err = StorageError("write failed")
        assert isinstance(err, PebbleError)
        assert "write failed" in str(err)


def test_get_config_success():
    """Loads config successfully."""
    mock_config = MagicMock()
    with patch("pebble.cli.load_config", return_value=mock_config):
        assert cli._get_config() == mock_config


def test_get_config_error():
    """Exits on config error."""
    with patch("pebble.cli.load_config", side_effect=FileNotFoundError("missing")):
        with pytest.raises(typer.Exit):
            cli._get_config()


def test_run_ollama_check_success():
    """Returns silently if Ollama reachable."""
    mock_config = MagicMock()
    mock_config.models.ollama_host = "http://localhost:11434"
    with patch("urllib.request.urlopen", return_value=MagicMock()):
        cli._run_ollama_check(mock_config)


def test_run_ollama_check_failure():
    """Exits if Ollama unreachable."""
    mock_config = MagicMock()
    mock_config.models.ollama_host = "http://localhost:11434"
    with patch("urllib.request.urlopen", side_effect=Exception("unreachable")):
        with pytest.raises(typer.Exit):
            cli._run_ollama_check(mock_config)


@patch("pebble.cli._get_config")
@patch("pebble.cli._run_ollama_check")
@patch("pebble.cli.asyncio.run")
@patch("pebble.cli.append_entry")
def test_log_command(mock_append, mock_run, mock_check, mock_config, tmp_path):
    """Log command orchestration."""
    cfg = MagicMock()
    cfg.storage.journal_dir = tmp_path
    mock_config.return_value = cfg

    mock_entry = MagicMock()
    mock_entry.mood = Mood.TENDER
    mock_entry.milestone_tags = []
    mock_run.return_value = mock_entry

    cli.log(note="test note", date_str=None, dry_run=False, verbose=False)

    mock_run.assert_called_once()
    mock_append.assert_called_once()


@patch("pebble.cli._get_config")
@patch("pebble.cli.iter_entries")
@patch("pebble.cli.Console.print")
def test_recent_command(mock_print, mock_iter, mock_config, tmp_path):
    """Recent command shows table."""
    cfg = MagicMock()
    cfg.storage.journal_dir = tmp_path
    mock_config.return_value = cfg

    mock_entry = MagicMock()
    mock_entry.date = date.today()
    mock_entry.mood = Mood.TENDER
    mock_entry.milestone_tags = []
    mock_entry.narrative = "narrative text"
    mock_iter.return_value = [mock_entry]

    cli.recent(weeks=1)
    mock_iter.assert_called_once()


@patch("pebble.cli._get_config")
@patch("pebble.cli.search_entries")
def test_search_command(mock_search, mock_config, tmp_path):
    """Search command calls search_entries."""
    cfg = MagicMock()
    cfg.storage.journal_dir = tmp_path
    mock_config.return_value = cfg
    mock_search.return_value = []

    cli.search(query="query text", tag=None, after=None, before=None)
    mock_search.assert_called_once()


@patch("pebble.cli._get_config")
@patch("pebble.cli.load_entry")
def test_view_command_not_found(mock_load, mock_config):
    """View command handles missing entry."""
    mock_config.return_value = MagicMock()
    mock_load.return_value = None

    with pytest.raises(typer.Exit):
        cli.view(entry_date_str=None)
