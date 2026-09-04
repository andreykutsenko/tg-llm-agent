"""Single entry point to the language model: text answers and tool-call steps."""

import dataclasses
import hashlib
import json
import logging
import re
import time

from bench.pricing import estimate_cost
from config import PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, Config, get_config
from observability import current_step, previous_payload_size, recorder, remember_payload
from llms import anthropic as anthropic_provider
from llms import ollama as ollama_provider
from llms.errors import LLMError, LLMTimeoutError
from llms.protocol import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    LLMResult,
    ToolCall,
    ToolSpec,
    Usage,
    assistant_message,
    tool_message,
    user_message,
)

__all__ = [
    "call_llm",
    "call_llm_step",
    "count_tokens",
    "strip_think_block",
    "LLMError",
    "LLMTimeoutError",
    "LLMResult",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "assistant_message",
    "tool_message",
    "user_message",
]

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Обрыв по лимиту токенов оставляет открытый <think> без закрывающего тега.
_THINK_UNCLOSED = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL | re.IGNORECASE)

_STEP_GENERATORS = {
    PROVIDER_OLLAMA: ollama_provider.generate_step,
    PROVIDER_ANTHROPIC: anthropic_provider.generate_step,
}
# Только Anthropic отдаёт счётчик токенов бесплатным эндпоинтом.
_TOKEN_COUNTERS = {
    PROVIDER_ANTHROPIC: anthropic_provider.count_tokens,
}
INCREMENT_BY_COUNT_TOKENS = "count_tokens"
INCREMENT_BY_CHARS = "chars"


def strip_think_block(text: str) -> str:
    """Drop reasoning blocks of models like qwen3; keep the text if nothing remains."""
    stripped = _THINK_BLOCK.sub("", text)
    stripped = _THINK_UNCLOSED.sub("", stripped).strip()
    return stripped if stripped else text.strip()


def _resolve(config: Config | None) -> tuple[Config, callable]:
    active_config = config if config is not None else get_config()
    generate_step = _STEP_GENERATORS.get(active_config.llm_provider)
    if generate_step is None:
        raise LLMError(f"Провайдер {active_config.llm_provider!r} не поддерживается.")
    return active_config, generate_step


BLOCK_SYSTEM = "system"
BLOCK_TOOLS = "tools"
BLOCK_USER = "user"
BLOCK_HASH_LENGTH = 16


def _block(kind: str, text: str) -> dict:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:BLOCK_HASH_LENGTH]
    return {"kind": kind, "chars": len(text), "hash": digest}


def _message_text(message: dict) -> str:
    text = message.get("content") or ""
    if message.get("role") == ROLE_ASSISTANT and message.get("tool_calls"):
        calls = [dataclasses.asdict(call) for call in message["tool_calls"]]
        text += json.dumps(calls, ensure_ascii=False, sort_keys=True)
    return text


def describe_context(
    system_prompt: str, tools: tuple[ToolSpec, ...], messages: list[dict]
) -> tuple[dict, ...]:
    """Size and fingerprint of every block of the input payload, without its content."""
    blocks = [_block(BLOCK_SYSTEM, system_prompt)]
    if tools:
        specs = [dataclasses.asdict(tool) for tool in tools]
        blocks.append(_block(BLOCK_TOOLS, json.dumps(specs, ensure_ascii=False, sort_keys=True)))
    blocks.extend(
        _block(message.get("role") or BLOCK_USER, _message_text(message)) for message in messages
    )
    return tuple(blocks)


async def count_tokens(
    messages: list[dict],
    tools: tuple[ToolSpec, ...] = (),
    config: Config | None = None,
    system: str | None = None,
) -> int | None:
    """Token count by the provider's tokenizer; None when the provider has no counter."""
    active_config = config if config is not None else get_config()
    counter = _TOKEN_COUNTERS.get(active_config.llm_provider)
    if counter is None:
        return None
    return await counter(messages, tools, active_config, system)


def _blank_tool_results(messages: list[dict]) -> list[dict]:
    return [
        {**message, "content": ""} if message.get("role") == ROLE_TOOL else message
        for message in messages
    ]


async def _count_increment(new_messages: list[dict], config: Config) -> tuple[int, int] | None:
    """(assistant tokens, tool-output tokens) of the appended pair, or None if uncountable."""
    if not config.llm_count_tokens or not new_messages:
        return None
    try:
        without_outputs = await count_tokens(_blank_tool_results(new_messages), (), config)
        with_outputs = await count_tokens(new_messages, (), config)
    except LLMError as error:
        logger.warning("Прирост контекста не посчитан через count_tokens: %s", error)
        return None
    if without_outputs is None or with_outputs is None:
        return None
    return (without_outputs, max(with_outputs - without_outputs, 0))


async def describe_increment(
    run_id: str, messages: list[dict], config: Config
) -> dict | None:
    """What this step added to the payload compared with the previous step of the run."""
    already_sent = previous_payload_size(run_id)
    remember_payload(run_id, len(messages))
    if already_sent == 0:
        return None
    new_messages = messages[already_sent:]
    assistant_chars = sum(
        len(_message_text(m)) for m in new_messages if m.get("role") == ROLE_ASSISTANT
    )
    tool_messages = [
        {"name": m.get("name", ""), "chars": len(_message_text(m))}
        for m in new_messages
        if m.get("role") == ROLE_TOOL
    ]
    counted = await _count_increment(new_messages, config)
    return {
        "method": INCREMENT_BY_COUNT_TOKENS if counted else INCREMENT_BY_CHARS,
        "assistant_chars": assistant_chars,
        "tool_chars": sum(item["chars"] for item in tool_messages),
        "assistant_tokens": counted[0] if counted else None,
        "tool_tokens": counted[1] if counted else None,
        "tool_messages": tool_messages,
    }


async def _record_step(
    config: Config, tools: tuple[ToolSpec, ...], messages: list[dict], usage: Usage
) -> None:
    position = current_step()
    if position is None:
        return
    increment = await describe_increment(position.run_id, messages, config)
    recorder.record_llm_call(
        run_id=position.run_id,
        step=position.step,
        model=config.llm_model,
        usage=usage,
        cost=estimate_cost(config.llm_model, usage),
        context_blocks=describe_context(config.system_prompt, tools, messages),
        increment=increment,
    )


async def call_llm_step(
    messages: list[dict],
    tools: tuple[ToolSpec, ...] = (),
    config: Config | None = None,
) -> LLMResult:
    """One step of the agent loop, in the same shape for every provider."""
    active_config, generate_step = _resolve(config)
    started = time.perf_counter()
    result = await generate_step(messages, tools, active_config)
    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = dataclasses.replace(result.usage, latency_ms=latency_ms)
    await _record_step(active_config, tools, messages, usage)
    return LLMResult(
        text=strip_think_block(result.text), tool_calls=result.tool_calls, usage=usage
    )


async def call_llm(prompt: str, config: Config | None = None) -> str:
    """Send a stateless request to the configured provider and return its answer."""
    result = await call_llm_step([user_message(prompt)], (), config)
    return result.text
