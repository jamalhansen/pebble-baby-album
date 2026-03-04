"""PydanticAI agents for journal structuring and photo description."""
from datetime import date
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Config
from .models import JournalEntry, PhotoDescription, WeeklySummary


def _make_ollama_model(model_name: str, ollama_host: str) -> OpenAIModel:
    """Ollama exposes an OpenAI-compatible API; point pydantic-ai's OpenAI model at it."""
    provider = OpenAIProvider(base_url=f"{ollama_host}/v1", api_key="ollama")  # Ollama ignores this; required by the client
    return OpenAIModel(model_name=model_name, provider=provider)


def model_from_name(name: str, config: Config) -> OpenAIModel:
    """Build an Ollama model from a bare model name string, using the configured host."""
    return _make_ollama_model(name, config.models.ollama_host)


def make_journal_agent(config: Config, model=None) -> Agent[None, JournalEntry]:
    """Create the journal structuring agent."""
    if model is None:
        model = _make_ollama_model(config.models.text_model, config.models.ollama_host)

    return Agent(
        model=model,
        result_type=JournalEntry,
        system_prompt=(
            "You are a baby journal assistant. You take raw, messy notes from a parent "
            "about their baby and structure them into a journal entry.\n\n"
            "Your job:\n"
            "1. Identify milestone categories from the note "
            "(motor-skills, social-emotional, cognitive, language, feeding, sleep, health, first)\n"
            "2. Detect the emotional mood of the entry "
            "(joyful, tender, proud, tired, worried, grateful, funny)\n"
            "3. Clean up the narrative while preserving the parent's voice. "
            "Do NOT rewrite into generic prose. Keep their words, fix grammar only where "
            "it's unclear, and organize into flowing sentences.\n"
            "4. If the note mentions something happening for the first time, "
            "ALWAYS include 'first' in milestone_tags.\n\n"
            "Return a structured JournalEntry with all fields filled in."
        ),
    )


def make_vision_agent(config: Config, model=None) -> Agent[None, PhotoDescription]:
    """Create the photo description agent (uses LLaVA via Ollama)."""
    if model is None:
        model = _make_ollama_model(config.models.vision_model, config.models.ollama_host)

    return Agent(
        model=model,
        result_type=PhotoDescription,
        system_prompt=(
            "You are describing a baby photo for a private journal. "
            "The description will be stored as searchable text alongside the photo.\n\n"
            "Describe what you see in detail:\n"
            "- The baby's position, expression, and what they're wearing\n"
            "- What they're interacting with (toys, people, objects)\n"
            "- The setting (lighting, room, surfaces)\n"
            "- Any notable developmental observations (reaching, gripping, eye tracking)\n\n"
            "Be warm but specific. These descriptions help the parent search and relive moments later."
        ),
    )


def make_summary_agent(config: Config, model=None) -> Agent[None, WeeklySummary]:
    """Create the weekly/monthly summary agent."""
    if model is None:
        model = _make_ollama_model(config.models.text_model, config.models.ollama_host)

    return Agent(
        model=model,
        result_type=WeeklySummary,
        system_prompt=(
            "You are a baby journal assistant creating a weekly summary. "
            "You will receive a collection of daily journal entries and should synthesize them "
            "into a warm, meaningful weekly summary.\n\n"
            "Your job:\n"
            "1. Identify the most notable highlights from the week\n"
            "2. List any milestones the baby reached\n"
            "3. Write a warm narrative summary that captures the week's emotional arc\n\n"
            "Be concise but warm. This is a keepsake the parent will re-read."
        ),
    )


async def log_entry(
    raw_text: str,
    entry_date: date,
    config: Config,
    model=None,
) -> JournalEntry:
    """Run the journal agent and return a structured JournalEntry."""
    agent = make_journal_agent(config, model=model)
    age_weeks = config.age_weeks(entry_date)
    prompt = (
        f"Baby's name: {config.baby.name}\n"
        f"Today's date: {entry_date.isoformat()}\n"
        f"Baby is {age_weeks} weeks old.\n\n"
        f"Parent's note:\n{raw_text}"
    )
    result = await agent.run(prompt)
    entry = result.data
    # Ensure date and age_weeks are set from config, not hallucinated
    entry.date = entry_date
    entry.age_weeks = age_weeks
    entry.raw_input = raw_text
    return entry


async def describe_photo(
    image_path: Path,
    config: Config,
    model=None,
) -> PhotoDescription:
    """Run the vision agent and return a PhotoDescription."""
    agent = make_vision_agent(config, model=model)

    # Load image as base64 for multimodal input
    import base64
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()
    suffix = image_path.suffix.lower().lstrip(".")
    mime_type = f"image/{suffix}" if suffix in ("jpg", "jpeg", "png", "gif", "webp") else "image/jpeg"

    from pydantic_ai.messages import BinaryContent

    result = await agent.run(
        [
            "Please describe this baby photo for the journal.",
            BinaryContent(data=image_bytes, media_type=mime_type),
        ]
    )
    photo = result.data
    photo.file_path = str(image_path)
    return photo


async def summarize_entries(
    entries: list[JournalEntry],
    week_start: date,
    week_end: date,
    config: Config,
    model=None,
) -> WeeklySummary:
    """Run the summary agent over a list of entries."""
    agent = make_summary_agent(config, model=model)

    entry_texts = []
    for e in entries:
        entry_texts.append(
            f"--- {e.date.isoformat()} (week {e.age_weeks}, mood: {e.mood.value}) ---\n{e.narrative}"
        )
    combined = "\n\n".join(entry_texts)

    prompt = (
        f"Baby's name: {config.baby.name}\n"
        f"Week: {week_start.isoformat()} to {week_end.isoformat()}\n\n"
        f"Journal entries:\n\n{combined}"
    )

    result = await agent.run(prompt)
    summary = result.data
    summary.week_start = week_start
    summary.week_end = week_end
    return summary
