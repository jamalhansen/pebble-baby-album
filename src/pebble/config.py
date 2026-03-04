from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


XDG_CONFIG_HOME = Path.home() / ".config"
DOCUMENTS_CONFIG_PATH = Path.home() / "Documents" / "pebble" / "config.toml"
USER_CONFIG_PATH = XDG_CONFIG_HOME / "pebble" / "config.toml"
REPO_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.toml"


def _find_config() -> Path:
    """
    Config lookup order:
    1. $PEBBLE_CONFIG env var
    2. ~/Documents/pebble/config.toml  (everything in one place)
    3. ~/.config/pebble/config.toml    (XDG standard)
    4. ./config.toml                   (repo root, for development)
    """
    import os
    env = os.environ.get("PEBBLE_CONFIG")
    if env:
        return Path(env)
    if DOCUMENTS_CONFIG_PATH.exists():
        return DOCUMENTS_CONFIG_PATH
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    return REPO_CONFIG_PATH


@dataclass
class BabyConfig:
    name: str
    birth_date: date


@dataclass
class ModelsConfig:
    text_model: str
    vision_model: str
    ollama_host: str


@dataclass
class StorageConfig:
    journal_dir: Path
    inbox_dir: Path
    processed_dir: Path


@dataclass
class WebConfig:
    port: int


@dataclass
class Config:
    baby: BabyConfig
    models: ModelsConfig
    storage: StorageConfig
    web: WebConfig

    def age_weeks(self, on_date: date) -> int:
        delta = on_date - self.baby.birth_date
        return max(0, delta.days // 7)


def load_config(path: Path | None = None) -> Config:
    config_path = path or _find_config()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found.\n\n"
            f"Simplest setup — put everything in one place:\n"
            f"  {DOCUMENTS_CONFIG_PATH}\n\n"
            f"Or use the XDG location:\n"
            f"  {USER_CONFIG_PATH}\n\n"
            f"Or set $PEBBLE_CONFIG to any path.\n\n"
            f"Minimal config:\n"
            f"  [baby]\n"
            f"  name = \"Baby\"\n"
            f"  birth_date = 2025-01-01\n"
        )
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    baby = raw.get("baby", {})
    models = raw.get("models", {})
    storage = raw.get("storage", {})
    web = raw.get("web", {})

    birth_date_raw = baby.get("birth_date", "2025-01-01")
    if isinstance(birth_date_raw, date):
        birth_date = birth_date_raw
    else:
        birth_date = date.fromisoformat(str(birth_date_raw))

    def _resolve_dir(raw, default: Path) -> Path:
        if raw is None:
            return default
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = config_path.parent / p
        return p

    journal_dir = _resolve_dir(
        storage.get("journal_dir"),
        Path.home() / "Documents" / "pebble",
    )
    inbox_dir = _resolve_dir(
        storage.get("inbox_dir"),
        journal_dir / "inbox",
    )
    processed_dir = _resolve_dir(
        storage.get("processed_dir"),
        inbox_dir / "processed",
    )

    return Config(
        baby=BabyConfig(
            name=baby.get("name", "Baby"),
            birth_date=birth_date,
        ),
        models=ModelsConfig(
            text_model=models.get("text_model", "qwen2.5:3b"),
            vision_model=models.get("vision_model", "llava:7b"),
            ollama_host=models.get("ollama_host", "http://localhost:11434"),
        ),
        storage=StorageConfig(
            journal_dir=journal_dir,
            inbox_dir=inbox_dir,
            processed_dir=processed_dir,
        ),
        web=WebConfig(
            port=web.get("port", 5555),
        ),
    )
