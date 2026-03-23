"""Agents for journal structuring and photo description, using standardized providers."""
import base64
import io
import time
from datetime import date
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener
from rich.console import Console

from local_first_common.providers.ollama import OllamaProvider
from local_first_common.tracking import register_tool, timed_run
from .config import Config
from .models import (
    EntryMetadata,
    JournalEntry,
    PhotoAnalysis,
    PhotoDescription,
    WeeklySummary,
)

register_heif_opener()

_console = Console(stderr=True)

_TOOL = register_tool("pebble")

# Vision models don't benefit from huge images; cap at this on the longest side.
_MAX_VISION_PX = 1024


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


def _get_provider(config: Config, model_name: str) -> OllamaProvider:
    return OllamaProvider(model=model_name)


async def log_entry(
    raw_text: str,
    entry_date: date,
    config: Config,
    model: str | None = None,
) -> JournalEntry:
    """Classify the parent's note (tags + mood) and build a JournalEntry."""
    model_name = model or config.models.text_model
    age_weeks = config.age_weeks(entry_date)
    prompt = (
        f"Baby's name: {config.baby.name}\n"
        f"Today's date: {entry_date.isoformat()}\n\n"
        f"Parent's note:\n{raw_text}"
    )
    
    llm = _get_provider(config, model_name)
    with timed_run("pebble", llm.model, source_location=entry_date.isoformat()) as run:
        meta_dict = await llm.acomplete(
            system=_JOURNAL_SYSTEM,
            user=prompt,
            response_model=EntryMetadata,
        )
        run.item_count = 1
        run.input_tokens = getattr(llm, "input_tokens", None) or None
        run.output_tokens = getattr(llm, "output_tokens", None) or None

    meta = EntryMetadata.model_validate(meta_dict)
    
    return JournalEntry(
        date=entry_date,
        age_weeks=age_weeks,
        milestone_tags=meta.milestone_tags,
        mood=meta.mood,
        raw_input=raw_text,
        narrative=raw_text,  # parent's exact words — no AI rewriting
    )


def _to_jpeg_bytes(image_path: Path) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """
    Return (jpeg_bytes, original_size, final_size).

    Resizes so the longest side is at most _MAX_VISION_PX before encoding.
    """
    with Image.open(image_path) as img:
        original_size = img.size
        img = img.convert("RGB")
        img.thumbnail((_MAX_VISION_PX, _MAX_VISION_PX), Image.LANCZOS)
        final_size = img.size
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), original_size, final_size


async def describe_photo(
    image_path: Path,
    config: Config,
    model: str | None = None,
) -> PhotoDescription:
    """Call the vision model and return a PhotoDescription."""
    model_name = model or config.models.vision_model

    t0 = time.monotonic()
    image_bytes, orig_size, final_size = _to_jpeg_bytes(image_path)
    t_convert = time.monotonic() - t0
    kb = len(image_bytes) / 1024
    _console.print(
        f"  [dim]convert: {orig_size[0]}×{orig_size[1]} → {final_size[0]}×{final_size[1]}"
        f"  |  {kb:.0f} KB  |  {t_convert:.2f}s[/]"
    )

    image_b64 = base64.b64encode(image_bytes).decode()

    _console.print(f"  [dim]calling {model_name} …[/]")
    t1 = time.monotonic()
    
    llm = _get_provider(config, model_name)
    with timed_run("pebble", llm.model, source_location=str(image_path)) as run:
        photo_dict = await llm.acomplete(
            system=_VISION_SYSTEM,
            user=f"Please describe this photo of {config.baby.name} for the journal.",
            response_model=PhotoAnalysis,
            images=[image_b64],
        )
        run.item_count = 1
        run.input_tokens = getattr(llm, "input_tokens", None) or None
        run.output_tokens = getattr(llm, "output_tokens", None) or None

    analysis = PhotoAnalysis.model_validate(photo_dict)
    
    t_llm = time.monotonic() - t1
    _console.print(f"  [dim]llm: {t_llm:.1f}s[/]")

    return PhotoDescription(
        file_path=str(image_path),
        description=analysis.description,
    )


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
    
    llm = _get_provider(config, model_name)
    with timed_run("pebble", llm.model, source_location=week_start.isoformat()) as run:
        summary_dict = await llm.acomplete(
            system=_SUMMARY_SYSTEM,
            user=prompt,
            response_model=WeeklySummary,
        )
        run.item_count = 1
        run.input_tokens = getattr(llm, "input_tokens", None) or None
        run.output_tokens = getattr(llm, "output_tokens", None) or None

    summary = WeeklySummary.model_validate(summary_dict)
    
    summary.week_start = week_start
    summary.week_end = week_end
    return summary
