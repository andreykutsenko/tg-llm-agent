import pytest
from conftest import FakeClient, anthropic_response, make_config, openai_response

import llms
from llms import anthropic as anthropic_provider
from llms import ollama as ollama_provider
from llms.errors import LLMTimeoutError


async def test_call_llm_returns_model_text(monkeypatch):
    client = FakeClient(response=openai_response("Ответ модели"))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: client)

    answer = await llms.call_llm("вопрос", config=make_config())

    assert answer == "Ответ модели"
    assert client.closed is True


async def test_think_block_is_stripped(monkeypatch):
    raw = "<think>рассуждаю долго\nи многострочно</think>\nКраткий ответ"
    client = FakeClient(response=openai_response(raw))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: client)

    answer = await llms.call_llm("вопрос", config=make_config())

    assert answer == "Краткий ответ"
    assert "<think>" not in answer


def test_think_only_answer_falls_back_to_original_text():
    raw = "<think>только рассуждение</think>"
    assert llms.strip_think_block(raw) == raw


async def test_request_contains_exactly_system_and_current_user_message(monkeypatch):
    client = FakeClient(response=openai_response("ok"))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: client)
    config = make_config(system_prompt="Системный промпт")

    await llms.call_llm("текущее сообщение", config=config)
    await llms.call_llm("следующее сообщение", config=config)

    for call, prompt in zip(client.calls, ("текущее сообщение", "следующее сообщение")):
        assert call["messages"] == [
            {"role": "system", "content": "Системный промпт"},
            {"role": "user", "content": prompt},
        ]


async def test_anthropic_provider_switches_client(monkeypatch):
    ollama_client = FakeClient(response=openai_response("локальный ответ"))
    anthropic_client = FakeClient(response=anthropic_response("облачный ответ"))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: ollama_client)
    monkeypatch.setattr(
        anthropic_provider, "AsyncAnthropic", lambda **kwargs: anthropic_client
    )

    local = await llms.call_llm("вопрос", config=make_config(llm_provider="ollama"))
    cloud = await llms.call_llm(
        "вопрос",
        config=make_config(
            llm_provider="anthropic", llm_model="claude-opus-5", llm_api_key="sk-ant-test"
        ),
    )

    assert local == "локальный ответ"
    assert cloud == "облачный ответ"
    assert len(ollama_client.calls) == 1
    assert len(anthropic_client.calls) == 1
    assert anthropic_client.calls[0]["model"] == "claude-opus-5"
    assert anthropic_client.calls[0]["system"] == make_config().system_prompt
    assert anthropic_client.calls[0]["messages"] == [
        {"role": "user", "content": "вопрос"}
    ]


async def test_timeout_is_translated_to_llm_timeout_error(monkeypatch):
    from openai import APITimeoutError

    try:  # openai 3.x ships httpx2, earlier versions ship httpx
        from httpx2 import Request
    except ImportError:  # pragma: no cover - depends on installed SDK version
        from httpx import Request

    request = Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    client = FakeClient(error=APITimeoutError(request=request))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: client)

    with pytest.raises(LLMTimeoutError):
        await llms.call_llm("вопрос", config=make_config())
    assert client.closed is True


def test_strip_think_block_handles_unclosed_tag():
    """Обрыв по лимиту токенов оставляет <think> без закрывающего тега."""
    text = "Ответ.\n<think>рассуждение оборвалось"
    assert llms.strip_think_block(text) == "Ответ."


def test_strip_think_block_keeps_text_when_only_unclosed_think():
    """Если кроме оборванного размышления ничего нет — отдаём исходный текст."""
    text = "<think>только размышление"
    assert llms.strip_think_block(text) == text.strip()
