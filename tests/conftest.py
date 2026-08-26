import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as config_module  # noqa: E402

ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_TOKENS",
    "SYSTEM_PROMPT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Isolated environment: no inherited variables, no reading of a real .env."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: False)
    config_module.get_config.cache_clear()
    yield monkeypatch
    config_module.get_config.cache_clear()


def make_config(**overrides):
    values = {
        "telegram_bot_token": "123456:test-token-value",
        "llm_provider": "ollama",
        "llm_base_url": "http://127.0.0.1:11434/v1",
        "llm_model": "qwen3:1.7b",
        "llm_api_key": "ollama",
        "llm_timeout_seconds": 180.0,
        "llm_max_tokens": 512,
        "system_prompt": "Отвечай по-русски, кратко и по существу.",
    }
    values.update(overrides)
    return config_module.Config(**values)


class FakeMessage:
    """Minimal stand-in for aiogram Message: records everything sent back."""

    def __init__(self, text="привет", chat_id=42):
        self.text = text
        self.chat = SimpleNamespace(id=chat_id)
        self.answers: list[str] = []
        self.chat_actions: list[str] = []
        message = self

        class _Bot:
            async def send_chat_action(self, chat_id, action):
                message.chat_actions.append(action)

        self.bot = _Bot()

    async def answer(self, text):
        self.answers.append(text)


def openai_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def anthropic_response(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    """Async client double that records the kwargs of the model call."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []
        self.closed = False
        client = self

        async def create(**kwargs):
            client.calls.append(kwargs)
            if client.error is not None:
                raise client.error
            return client.response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.messages = SimpleNamespace(create=create)

    async def close(self):
        self.closed = True
