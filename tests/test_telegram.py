import pytest
from conftest import FakeMessage, make_config

import bot
from harness import AgentResult
from llms.errors import LLMError, LLMTimeoutError


@pytest.fixture
def config(monkeypatch):
    """Handlers read the configuration themselves — give them a predictable one."""
    value = make_config()
    monkeypatch.setattr(bot, "get_config", lambda: value)
    return value


def answers_with(monkeypatch, text):
    async def fake_run_agent(user_text, config):
        return AgentResult(text=text, steps=1, limit_reached=False)

    monkeypatch.setattr(bot, "run_agent", fake_run_agent)


def fails_with(monkeypatch, error):
    async def fake_run_agent(user_text, config):
        raise error

    monkeypatch.setattr(bot, "run_agent", fake_run_agent)


def test_long_answer_is_split_under_telegram_limit():
    text = ("слово " * 2000).strip()
    assert len(text) >= 10000

    parts = bot.split_message(text)

    assert len(parts) > 1
    assert all(len(part) <= bot.TELEGRAM_MESSAGE_LIMIT for part in parts)
    assert "".join(part.replace(" ", "") for part in parts) == text.replace(" ", "")


def test_split_happens_on_line_break_not_inside_word():
    line = "а" * 1000
    text = "\n".join([line] * 12)

    parts = bot.split_message(text)

    assert len(parts) > 1
    for part in parts:
        assert len(part) <= bot.TELEGRAM_MESSAGE_LIMIT
        for chunk in part.split("\n"):
            assert chunk == line


def test_split_falls_back_to_hard_cut_without_separators():
    text = "я" * 10000

    parts = bot.split_message(text)

    assert [len(part) for part in parts] == [4096, 4096, 1808]
    assert "".join(parts) == text


async def test_empty_model_answer_gives_explicit_message(monkeypatch, config):
    answers_with(monkeypatch, "   \n  ")
    message = FakeMessage()

    await bot.handle_text(message)

    assert message.answers == [bot.EMPTY_ANSWER_TEXT]


async def test_timeout_is_reported_to_user_without_raising(monkeypatch, config):
    fails_with(monkeypatch, LLMTimeoutError("Модель qwen3:1.7b не ответила за 180 секунд."))
    message = FakeMessage()

    await bot.handle_text(message)

    assert len(message.answers) == 1
    assert "не ответила" in message.answers[0]
    assert "Traceback" not in message.answers[0]


async def test_connection_error_is_reported_with_hint(monkeypatch, config):
    fails_with(monkeypatch, LLMError("Не удалось подключиться к http://127.0.0.1:11434/v1."))
    message = FakeMessage()

    await bot.handle_text(message)

    assert "LLM_BASE_URL" in message.answers[0]


async def test_unexpected_error_does_not_escape_handler(monkeypatch, config):
    fails_with(monkeypatch, RuntimeError("что угодно"))
    message = FakeMessage()

    await bot.handle_text(message)

    assert message.answers == [bot.GENERIC_ERROR_TEXT]


async def test_typing_action_is_shown_before_model_call(monkeypatch, config):
    answers_with(monkeypatch, "ответ")
    message = FakeMessage()

    await bot.handle_text(message)

    assert message.chat_actions == ["typing"]
    assert message.answers == ["ответ"]


async def test_non_text_message_gets_polite_answer(config):
    message = FakeMessage(text=None)

    await bot.handle_unsupported(message)

    assert message.answers == [bot.UNSUPPORTED_CONTENT_TEXT]


async def test_message_from_unlisted_id_is_ignored(monkeypatch, caplog):
    allowed = make_config(telegram_allowed_ids=(100, 200))
    monkeypatch.setattr(bot, "get_config", lambda: allowed)
    answers_with(monkeypatch, "ответ")
    message = FakeMessage(text="привет", user_id=999)

    await bot.handle_text(message)

    assert message.answers == []
    assert message.chat_actions == []
    assert "999" in caplog.text


async def test_message_from_listed_id_is_answered(monkeypatch):
    allowed = make_config(telegram_allowed_ids=(100, 200))
    monkeypatch.setattr(bot, "get_config", lambda: allowed)
    answers_with(monkeypatch, "ответ")
    message = FakeMessage(text="привет", user_id=200)

    await bot.handle_text(message)

    assert message.answers == ["ответ"]
