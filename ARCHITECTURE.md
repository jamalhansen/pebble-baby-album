# Privacy-First Baby Journal (pebble)

## What This Is

A local-first CLI tool and web viewer that turns quick, messy notes about a baby into structured, searchable journal entries. Photos are described by a local vision model. Everything stays on-device. Nothing is ever uploaded anywhere.

## Core Principles

1. **Privacy is non-negotiable.** No network calls, no cloud APIs, no telemetry. All inference runs through a local Ollama instance.
2. **Input should be fast and messy.** The user types quick notes like a text message. The model handles structuring.
3. **Output is plain markdown.** YAML frontmatter for structured data, prose body for the narrative. Files are human-readable and Obsidian-compatible.
4. **Small models only.** Target Qwen 2.5 3B for text and LLaVA 7B for vision. If it needs a bigger model, the design is wrong.

## Tech Stack

- **Python 3.11+** with **uv** for package management
- **Typer** for the CLI
- **PydanticAI** for structured LLM output (type-safe, Ollama-compatible via OpenAI-compatible endpoint)
- **Ollama** as the local inference backend
- **Flask** with Jinja2 for the optional read-only web viewer
- **python-frontmatter** for reading/writing YAML+markdown files
- **Pillow** for EXIF date extraction from photos
- Markdown files as the only storage layer (no database)

## Architecture

```
CLI Input (text/photo/inbox) -> PydanticAI Agent -> Pydantic Model -> Markdown File
                                                                            |
                                                                            v
                                                               Web Viewer (Flask)
                                                               reads markdown files
                                                               renders timeline
```

There are three PydanticAI agents:

1. **journal_agent** — Takes raw text input, returns a `JournalEntry` with date, age_weeks, milestone_tags, mood, and a cleaned-up narrative. Preserves the parent's voice.
2. **vision_agent** — Takes an image path, sends it to LLaVA via Ollama, returns a `PhotoDescription` with a rich, detailed description of what's in the photo.
3. **summary_agent** — Takes a list of `JournalEntry` objects and a date range, returns a `WeeklySummary` with highlights, milestones, and narrative prose.

### Ollama Integration

pydantic-ai v1.x has no `OllamaModel`. Ollama is accessed via its OpenAI-compatible endpoint:

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

def _make_ollama_model(model_name: str, ollama_host: str) -> OpenAIModel:
    provider = OpenAIProvider(base_url=f"{ollama_host}/v1", api_key="ollama")
    return OpenAIModel(model_name=model_name, provider=provider)
```

## Data Models

```python
from pydantic import BaseModel
from datetime import date
from enum import Enum

class MilestoneTag(str, Enum):
    MOTOR_SKILLS = "motor-skills"
    SOCIAL_EMOTIONAL = "social-emotional"
    COGNITIVE = "cognitive"
    LANGUAGE = "language"
    FEEDING = "feeding"
    SLEEP = "sleep"
    HEALTH = "health"
    FIRST = "first"

class Mood(str, Enum):
    JOYFUL = "joyful"
    TENDER = "tender"
    PROUD = "proud"
    TIRED = "tired"
    WORRIED = "worried"
    GRATEFUL = "grateful"
    FUNNY = "funny"

MOOD_EMOJI = {Mood.JOYFUL: "😄", Mood.TENDER: "🥹", ...}

class PhotoDescription(BaseModel):
    file_path: str
    description: str
    taken_at: date | None = None

class JournalEntry(BaseModel):
    date: date
    age_weeks: int
    milestone_tags: list[MilestoneTag]
    mood: Mood
    raw_input: str
    narrative: str
    photos: list[PhotoDescription] = []

class WeeklySummary(BaseModel):
    week_start: date
    week_end: date
    highlights: list[str]
    milestones_reached: list[str]
    narrative: str
```

## Directory Structure

```
baby-journal/                    # Repo — code only, no personal data
├── ARCHITECTURE.md
├── pyproject.toml
├── config.toml                  # Dev fallback only — gitignored
├── src/
│   └── pebble/
│       ├── __init__.py
│       ├── cli.py               # Typer CLI (entry point)
│       ├── models.py            # Pydantic models
│       ├── agents.py            # PydanticAI agents (journal, vision, summary)
│       ├── storage.py           # Read/write/search markdown files
│       ├── inbox.py             # Photo inbox batch processing
│       ├── summary.py           # Weekly/monthly summary generation
│       └── config.py            # Config loading, age_weeks computation
│   └── web/
│       ├── __init__.py
│       ├── app.py               # Flask app (read-only)
│       ├── templates/
│       │   ├── base.html
│       │   ├── index.html       # Timeline view
│       │   ├── entry.html       # Single entry view
│       │   └── milestones.html  # Firsts timeline
│       └── static/
│           └── style.css
└── tests/
    ├── test_agents.py
    ├── test_storage.py
    ├── test_inbox.py
    └── fixtures/                # Sample markdown entries for testing

~/Documents/pebble/              # User data — lives outside the repo
├── config.toml                  # Primary config location
├── YYYY-MM-DD.md                # One journal entry file per day
├── summaries/                   # Generated weekly/monthly summaries
└── inbox/
    ├── (drop photos here)
    └── processed/
        └── YYYY-MM-DD/          # Photos moved here after inbox processing
```

## Configuration

Config lookup order (first found wins):

1. `$PEBBLE_CONFIG` env var
2. `~/Documents/pebble/config.toml` ← primary user location
3. `~/.config/pebble/config.toml` ← XDG standard
4. `./config.toml` ← repo root (dev only, gitignored)

```toml
[baby]
name = "Baby"
birth_date = 2025-01-01          # Used to compute age_weeks automatically

[models]
text_model = "qwen2.5:3b"        # Ollama model for text structuring
vision_model = "llava:7b"        # Ollama model for photo descriptions
ollama_host = "http://localhost:11434"

# Storage defaults — uncomment to override
# [storage]
# journal_dir   = "~/Documents/pebble"
# inbox_dir     = "~/Documents/pebble/inbox"
# processed_dir = "~/Documents/pebble/inbox/processed"

[web]
port = 5555
```

`config.py` provides a `Config` dataclass with `age_weeks(on_date: date) -> int` computed from `birth_date`.

## CLI Commands

The binary is `pebble`. All commands have `--config / -c` to override the config path.

### `pebble log`

```bash
pebble log "grabbed my finger at 2am, big smiles at the ceiling fan"
pebble log                           # opens $EDITOR
echo "first real laugh" | pebble log # pipe from stdin
pebble log "note" --dry-run          # print result without saving
pebble log "note" --model qwen2.5:7b # override model for this run
```

Sends text to `journal_agent`, appends structured result to `~/Documents/pebble/YYYY-MM-DD.md`.

### `pebble photo`

```bash
pebble photo ~/Desktop/morning-smile.jpg
pebble photo ~/Desktop/morning-smile.jpg --note "first real laugh"
pebble photo ~/Desktop/morning-smile.jpg --model llava:13b
pebble photo ~/Desktop/morning-smile.jpg --dry-run
```

Sends image to `vision_agent` (LLaVA), appends a `## Photos` section to the entry. Photos are never copied — only the path is stored.

### `pebble inbox`

```bash
pebble inbox              # describe and file all photos, move to processed/
pebble inbox --dry-run    # preview without making any changes
pebble inbox --verbose    # print each photo's description as it's generated
pebble inbox --model moondream   # use a lighter vision model
```

Batch-processes all images in `inbox_dir`. Each photo is dated from EXIF `DateTimeOriginal` (falling back to mtime), described by `vision_agent`, appended to the correct day's journal, and moved to `processed/YYYY-MM-DD/`. Filename collisions are handled with a counter suffix (`photo-1.jpg`, `photo-2.jpg`, etc.). Errors on individual photos are non-fatal — logged and counted as skipped.

### `pebble recent`

```bash
pebble recent              # last 7 days
pebble recent --weeks 4    # last 4 weeks
```

Compact terminal timeline: date, week number, mood emoji, tags, narrative preview.

### `pebble search`

```bash
pebble search "smile"
pebble search --tag motor-skills
pebble search --tag first --after 2025-12-01
```

Tag search reads YAML frontmatter. Text search is case-insensitive grep through file contents.

### `pebble view`

```bash
pebble view                # today
pebble view 2025-12-15     # specific date
```

Pretty-prints a single day's entry with full narrative, tags, mood, and photo descriptions.

### `pebble summary`

```bash
pebble summary --week
pebble summary --month
pebble summary --week --dry-run          # print without saving
pebble summary --week --model qwen2.5:7b
```

Reads all entries in the date range, sends them to `summary_agent`, writes result to `journal_dir/summaries/`. Note: `--month` has no short flag; `-m` is reserved for `--model`.

### `pebble serve`

```bash
pebble serve               # http://localhost:5555
pebble serve --port 8080
```

Read-only Flask web viewer. Serves only to `127.0.0.1`. Zero external network requests.

## Markdown File Format

One file per day at `~/Documents/pebble/YYYY-MM-DD.md`:

```markdown
---
date: 2025-12-15
age_weeks: 4
milestone_tags:
  - motor-skills
  - feeding
  - first
mood: proud
raw_input: grabbed my finger at 2am
---

Her grip strength is wild. She latched onto my thumb during a bottle feeding
and I swear she was trying to pull the bottle closer herself.

## Photos

### ~/photos/baby-smile.jpg

Baby in a white onesie on a nursing pillow, right hand wrapped tightly around
an adult's index finger...
```

Multiple entries per day are appended with `---` separators. On append, frontmatter is merged: `milestone_tags` are unioned, `mood` is updated to the latest entry's mood.

## Web Viewer

Flask app reads markdown files from `journal_dir` and renders them in a timeline view. Read-only, serves to localhost only.

### Routes

- `GET /` — Timeline view, most recent first. Date, mood, tags, narrative preview.
- `GET /entry/<date>` — Full single-day view with photos section.
- `GET /milestones` — Entries tagged `first`, displayed as a visual timeline.
- `GET /search?q=<query>&tag=<tag>` — Search results.

## Testing

```bash
uv run pytest
```

Tests use mocked agents — no real Ollama calls needed.

- `test_agents.py` — Mocks via `AsyncMock`; verifies structured output, prompt content, model override
- `test_storage.py` — File I/O with `tmp_path`; verifies frontmatter parsing, append, tag merging
- `test_inbox.py` — Mocks Pillow EXIF + `describe_photo` + `append_entry`; verifies batch processing, dry-run, collision handling, error recovery

Fixtures in `tests/fixtures/` — 5 sample markdown entries covering different moods, tags, and photo sections.

## Key Pitfalls

- **Do not call any external API.** All inference goes through `localhost:11434` only.
- **Do not use a database.** Markdown files are the only storage layer.
- **Do not over-structure the narrative.** The model should preserve the parent's casual voice.
- **age_weeks must be computed, not guessed.** Always derive from `config.baby.birth_date` and the entry date.
- **Handle missing Ollama gracefully.** Check reachability before running inference; print a clear error with `ollama serve` hint.
- **Photos in `pebble photo` are never moved.** Only the path is stored. The inbox command is the exception — it explicitly moves photos to `processed/` after filing.
- **Frontmatter merging on append.** When adding to an existing day, `milestone_tags` = union of old and new; `mood` = latest.
- **pydantic-ai has no OllamaModel.** Use `OpenAIModel` + `OpenAIProvider` pointing at `{ollama_host}/v1`.
