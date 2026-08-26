"""Reading and validation of environment configuration."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

PROVIDER_OLLAMA = "ollama"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = (PROVIDER_OLLAMA, PROVIDER_ANTHROPIC)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_SYSTEM_PROMPT = "Отвечай по-русски, кратко и по существу."


class ConfigError(Exception):
    """Environment is missing or contains an invalid value."""


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_seconds: float
    llm_max_tokens: int
    system_prompt: str


def mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}***{secret[-4:]}"


def _read_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Переменная окружения {name} не задана. "
            f"Скопируйте .env.example в .env и заполните {name}."
        )
    return value


def _read_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", PROVIDER_OLLAMA).strip().lower() or PROVIDER_OLLAMA
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER={provider!r} не поддерживается. "
            f"Допустимые значения: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return provider


def _read_positive_number(name: str, default: float, converter) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = converter(raw)
    except ValueError as error:
        raise ConfigError(f"Переменная окружения {name}={raw!r} не является числом.") from error
    if value <= 0:
        raise ConfigError(f"Переменная окружения {name}={raw!r} должна быть больше нуля.")
    return value


def _read_api_key(provider: str) -> str:
    explicit_key = os.getenv("LLM_API_KEY", "").strip()
    if explicit_key:
        return explicit_key
    if provider == PROVIDER_ANTHROPIC:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ConfigError(
                "Для LLM_PROVIDER=anthropic нужен ключ: задайте LLM_API_KEY или ANTHROPIC_API_KEY."
            )
        return key
    return PROVIDER_OLLAMA


def _default_model(provider: str) -> str:
    if provider == PROVIDER_ANTHROPIC:
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_OLLAMA_MODEL


def load_config() -> Config:
    """Read configuration from .env and the process environment."""
    load_dotenv(override=False)
    provider = _read_provider()
    return Config(
        telegram_bot_token=_read_required("TELEGRAM_BOT_TOKEN"),
        llm_provider=provider,
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        llm_model=os.getenv("LLM_MODEL", "").strip() or _default_model(provider),
        llm_api_key=_read_api_key(provider),
        llm_timeout_seconds=_read_positive_number(
            "LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, float
        ),
        llm_max_tokens=int(
            _read_positive_number("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS, lambda raw: int(raw))
        ),
        system_prompt=os.getenv("SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT,
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide configuration, read once."""
    return load_config()


def describe_config(config: Config) -> str:
    """Human-readable configuration dump with secrets masked."""
    return (
        f"provider={config.llm_provider} "
        f"model={config.llm_model} "
        f"base_url={config.llm_base_url if config.llm_provider == PROVIDER_OLLAMA else '-'} "
        f"timeout={config.llm_timeout_seconds}s "
        f"max_tokens={config.llm_max_tokens} "
        f"telegram_token={mask_secret(config.telegram_bot_token)} "
        f"llm_api_key={mask_secret(config.llm_api_key)}"
    )
