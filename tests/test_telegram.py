from conftest import FakeMessage

import bot
from llms.errors import LLMError, LLMTimeoutError


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


async def test_empty_model_answer_gives_explicit_message(monkeypatch):
    async def fake_call_llm(prompt):
        return "   \n  "

    monkeypatch.setattr(bot, "call_llm", fake_call_llm)
    message = FakeMessage()

    await bot.handle_text(message)

    assert message.answers == [bot.EMPTY_ANSWER_TEXT]


async def test_timeout_is_reported_to_user_without_raising(monkeypatch):
    async def fake_call_llm(prompt):
        raise LLMTimeoutError("Модель qwen3:1.7b не ответила за 180 секунд.")

    monkeypatch.setattr(bot, "call_llm", fake_call_llm)
    message = FakeMessage()

    await bot.handle_text(message)

    assert len(message.answers) == 1
    assert "не ответила" in message.answers[0]
    assert "Traceback" not in message.answers[0]


async def test_connection_error_is_reported_with_hint(monkeypatch):
    async def fake_call_llm(prompt):
        raise LLMError("Не удалось подключиться к http://127.0.0.1:11434/v1.")

    monkeypatch.setattr(bot, "call_llm", fake_call_llm)
    message = FakeMessage()

    await bot.handle_text(message)

    assert "LLM_BASE_URL" in message.answers[0]


async def test_unexpected_error_does_not_escape_handler(monkeypatch):
    async def fake_call_llm(prompt):
        raise RuntimeError("что угодно")

    monkeypatch.setattr(bot, "call_llm", fake_call_llm)
    message = FakeMessage()

    await bot.handle_text(message)

    assert message.answers == [bot.GENERIC_ERROR_TEXT]


async def test_typing_action_is_shown_before_model_call(monkeypatch):
    async def fake_call_llm(prompt):
        return "ответ"

    monkeypatch.setattr(bot, "call_llm", fake_call_llm)
    message = FakeMessage()

    await bot.handle_text(message)

    assert message.chat_actions == ["typing"]
    assert message.answers == ["ответ"]


async def test_non_text_message_gets_polite_answer():
    message = FakeMessage(text=None)

    await bot.handle_unsupported(message)

    assert message.answers == [bot.UNSUPPORTED_CONTENT_TEXT]
