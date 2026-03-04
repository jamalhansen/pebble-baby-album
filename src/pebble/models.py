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


MOOD_EMOJI = {
    Mood.JOYFUL: "😄",
    Mood.TENDER: "🥹",
    Mood.PROUD: "🌟",
    Mood.TIRED: "😴",
    Mood.WORRIED: "😟",
    Mood.GRATEFUL: "🙏",
    Mood.FUNNY: "😂",
}


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
