# pebble — Privacy-First Baby Journal

A local-first CLI tool and web viewer that turns quick, messy notes into structured, searchable journal entries. Photos are described by a local vision model. **Everything stays on your device. Nothing is ever uploaded anywhere.**

---

## What It Does

- **Log entries** from the command line — a quick note, piped text, or your `$EDITOR`
- **Describe photos** using a local vision model (LLaVA via Ollama)
- **Structure entries** into tagged, mood-labelled markdown files using a local text model (Qwen 2.5 via Ollama)
- **Photo inbox** — drop photos in a folder, pebble batch-processes them using EXIF dates
- **Search** entries by text or milestone tag
- **View** a timeline in your browser, or read entries in the terminal
- **Generate summaries** of weekly or monthly progress

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management
- [Ollama](https://ollama.ai) running locally with:
  - `ollama pull qwen2.5:3b` (~2 GB, text structuring)
  - `ollama pull llava:7b` (~4.7 GB, photo descriptions)

---

## Getting Started

**1. Clone and install**
```bash
git clone https://github.com/jamalhansen/pebble-baby-album
cd pebble-baby-album
uv sync
uv pip install -e .
```

**2. Create your config**
```bash
mkdir -p ~/Documents/pebble
cp config.toml ~/Documents/pebble/config.toml
```

Edit `~/Documents/pebble/config.toml` — set your baby's name and birth date. That's the only required change.

> Config lookup order: `$PEBBLE_CONFIG` env var → `~/Documents/pebble/config.toml` → `~/.config/pebble/config.toml` → `./config.toml`

**3. Pull Ollama models** (one-time)
```bash
ollama pull qwen2.5:3b
ollama pull llava:7b
```

**4. Start Ollama**
```bash
ollama serve
```

**5. Log your first entry**
```bash
pebble log "she smiled at me for the first time today"
```

Entries are saved to `~/Documents/pebble/`.

---

## Configuration

Config lives at `~/Documents/pebble/config.toml` (or `~/.config/pebble/config.toml`, or set `$PEBBLE_CONFIG`):

```toml
[baby]
name = "Baby"
birth_date = 2025-01-01      # used to compute age in weeks automatically

[models]
text_model = "qwen2.5:3b"    # Ollama model for text structuring
vision_model = "llava:7b"    # Ollama model for photo descriptions
ollama_host = "http://localhost:11434"

# Storage defaults — uncomment to override
# [storage]
# journal_dir   = "~/Documents/pebble"
# inbox_dir     = "~/Documents/pebble/inbox"
# processed_dir = "~/Documents/pebble/inbox/processed"

[web]
port = 5555
```

---

## Usage

### Log a note

```bash
pebble log "grabbed my finger at 2am, big smiles at the ceiling fan"

# Open $EDITOR for a longer entry
pebble log

# Pipe from stdin
echo "first real laugh today during tummy time" | pebble log

# Dry run — print result without saving
pebble log "test note" --dry-run

# Use a different model for this run
pebble log "note" --model qwen2.5:7b
```

### Add a photo

```bash
pebble photo ~/Desktop/morning-smile.jpg
pebble photo ~/Desktop/morning-smile.jpg --note "first real laugh"
pebble photo ~/Desktop/morning-smile.jpg --model llava:13b
```

### Photo inbox (batch processing)

Drop photos into `~/Documents/pebble/inbox/` and run:

```bash
pebble inbox              # describe and file all photos, then move to processed/
pebble inbox --dry-run    # preview what would happen without making any changes
pebble inbox --verbose    # print each photo's description as it's generated
pebble inbox --model moondream   # use a lighter vision model
```

Each photo is dated using its **EXIF DateTimeOriginal** (falling back to file modification time), so photos taken on different days automatically land in the right journal entry. Processed photos are moved to `~/Documents/pebble/inbox/processed/YYYY-MM-DD/`.

### View recent entries

```bash
pebble recent              # last 7 days
pebble recent --weeks 4    # last 4 weeks
```

### View a single day

```bash
pebble view                # today
pebble view 2025-12-15     # specific date
```

### Search

```bash
pebble search "smile"
pebble search --tag motor-skills
pebble search --tag first --after 2025-12-01
```

Valid tags: `motor-skills`, `social-emotional`, `cognitive`, `language`, `feeding`, `sleep`, `health`, `first`

### Generate a summary

```bash
pebble summary --week
pebble summary --month
pebble summary --week --dry-run    # print without saving
pebble summary --week --model qwen2.5:7b
```

### Web viewer

```bash
pebble serve               # http://localhost:5555
pebble serve --port 8080
```

---

## Journal Format

Each day is a single markdown file at `~/Documents/pebble/YYYY-MM-DD.md`:

```markdown
---
date: 2025-12-15
age_weeks: 4
milestone_tags:
  - motor-skills
  - first
mood: proud
raw_input: grabbed my finger at 2am
---

Her grip strength is wild. She latched onto my thumb during a bottle feeding
and I swear she was trying to pull the bottle closer herself.

## Photos

### ~/photos/morning-smile.jpg

Baby in a white onesie on a nursing pillow, right hand wrapped tightly around
an adult's index finger...
```

Multiple entries per day are separated with `---`. Frontmatter is merged on append (tags are unioned, mood is updated to the latest entry's mood).

---

## Project Structure

```
pebble-baby-album/
├── src/pebble/
│   ├── cli.py               # Typer CLI (pebble command)
│   ├── models.py            # Pydantic data models
│   ├── agents.py            # Ollama-backed structuring agents
│   ├── storage.py           # Read/write/search markdown files
│   ├── inbox.py             # Photo inbox batch processing
│   ├── summary.py           # Weekly/monthly summary generation
│   ├── config.py            # Config loading
│   └── web/
│       ├── app.py           # Flask web viewer (read-only)
│       ├── templates/       # Jinja2 HTML templates
│       └── static/          # Stylesheet
└── tests/
    ├── test_storage.py
    ├── test_agents.py
    ├── test_inbox.py
    └── fixtures/            # Sample entries for testing
```

---

## Testing

```bash
uv run pytest
```

Tests use mocked agents — no real Ollama calls needed.

---

## Privacy

- Zero network calls. All inference is via `localhost:11434` (Ollama).
- No database. Markdown files only.
- Config and journal data live in your home directory, outside the repo.
- Photos are never copied — only the file path is stored (except inbox, which moves them to `processed/`).
- The web viewer is read-only and serves only to `127.0.0.1`.
