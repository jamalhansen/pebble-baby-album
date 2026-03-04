"""Agents for journal structuring and photo description, using Ollama's native API."""
import base64
import json
from datetime import date
from pathlib import Path

from ollama import AsyncClient

from .config import Config
from .models import EntryMetadata, JournalEntry, PhotoDescription, WeeklySummary


def model_from_name(name: str, config: Config) -> str:
    """Return the model name string (used when --model overrides the config default)."""
    return name


_JOURNAL_SYSTEM = (
    "You are a baby journal classifier. Given a parent's note about their baby, "
    "return ONLY two things:\n\n"
    "1. milestone_tags: which categories apply "
    "(motor-skills, social-emotional, cognitive, language, feeding, sleep, health, first). "
    "Only include tags clearly supported by the note. "
    "If something is described as happening for the first time, always include 'first'.\n\n"
    "2. mood: the emotional tone of the note "
    "(joyful, tender, proud, tired, worried, grateful, funny).\n\n"
    "Do NOT write or rewrite any narrative. Do NOT add information. Classify only."
)

_VISION_SYSTEM = (
    "You are describing a baby photo for a private journal. "
    "Write a brief description of at most 50 words. "
    "Describe what the baby is doing and where. Use the baby's name, not 'the baby'. "
    "Be factual and specific. No flowery language."
)

_SUMMARY_SYSTEM = (
    "You are a baby journal assistant creating a weekly summary. "
    "You will receive a collection of daily journal entries and should synthesize them "
    "into a warm, meaningful weekly summary.\n\n"
    "Your job:\n"
    "1. Identify the most notable highlights from the week\n"
    "2. List any milestones the baby reached\n"
    "3. Write a warm narrative summary that captures the week's emotional arc\n\n"
    "Be concise but warm. This is a keepsake the parent will re-read."
)


async def log_entry(
    raw_text: str,
    entry_date: date,
    config: Config,
    model: str | None = None,
) -> JournalEntry:
    """Classify the parent's note (tags + mood) and build a JournalEntry.

    The narrative is the parent's own words — the model only classifies, never writes.
    """
    model_name = model or config.models.text_model
    age_weeks = config.age_weeks(entry_date)
    prompt = (
        f"Baby's name: {config.baby.name}\n"
        f"Today's date: {entry_date.isoformat()}\n\n"
        f"Parent's note:\n{raw_text}"
    )
    client = AsyncClient(host=config.models.ollama_host)
    response = await client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": _JOURNAL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        format=EntryMetadata.model_json_schema(),
    )
    data = json.loads(response.message.content)
    meta = EntryMetadata.model_validate(data)
    return JournalEntry(
        date=entry_date,
        age_weeks=age_weeks,
        milestone_tags=meta.milestone_tags,
        mood=meta.mood,
        raw_input=raw_text,
        narrative=raw_text,  # parent's exact words — no AI rewriting
    )


async def describe_photo(
    image_path: Path,
    config: Config,
    model: str | None = None,
) -> PhotoDescription:
    """Call the vision model and return a PhotoDescription."""
    model_name = model or config.models.vision_model
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()

    client = AsyncClient(host=config.models.ollama_host)
    response = await client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": f"Please describe this photo of {config.baby.name} for the journal.",
                "images": [image_b64],
            },
        ],
        format=PhotoDescription.model_json_schema(),
    )
    data = json.loads(response.message.content)
    photo = PhotoDescription.model_validate(data)
    photo.file_path = str(image_path)
    return photo


async def summarize_entries(
    entries: list[JournalEntry],
    week_start: date,
    week_end: date,
    config: Config,
    model: str | None = None,
) -> WeeklySummary:
    """Call the summary model over a list of entries."""
    model_name = model or config.models.text_model
    entry_texts = [
        f"--- {e.date.isoformat()} (week {e.age_weeks}, mood: {e.mood.value}) ---\n{e.narrative}"
        for e in entries
    ]
    prompt = (
        f"Baby's name: {config.baby.name}\n"
        f"Week: {week_start.isoformat()} to {week_end.isoformat()}\n\n"
        f"Journal entries:\n\n" + "\n\n".join(entry_texts)
    )
    client = AsyncClient(host=config.models.ollama_host)
    response = await client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        format=WeeklySummary.model_json_schema(),
    )
    data = json.loads(response.message.content)
    summary = WeeklySummary.model_validate(data)
    summary.week_start = week_start
    summary.week_end = week_end
    return summary
