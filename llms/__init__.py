"""Single entry point to the language model: text answers and tool-call steps."""

import dataclasses
import hashlib
import json
import re
import time

from bench.pricing import estimate_cost
from config import PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, Config, get_config
from observability import current_step
from observability import recorder
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

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Обрыв по лимиту токенов оставляет открытый <think> без закрывающего тега.
_THINK_UNCLOSED = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL | re.IGNORECASE)

_STEP_GENERATORS = {
    PROVIDER_OLLAMA: ollama_provider.generate_step,
    PROVIDER_ANTHROPIC: anthropic_provider.generate_step,
}


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


def _record_step(
    config: Config, tools: tuple[ToolSpec, ...], messages: list[dict], usage: Usage
) -> None:
    position = current_step()
    if position is None:
        return
    recorder.record_llm_call(
        run_id=position.run_id,
        step=position.step,
        model=config.llm_model,
        usage=usage,
        cost=estimate_cost(config.llm_model, usage),
        context_blocks=describe_context(config.system_prompt, tools, messages),
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
    _record_step(active_config, tools, messages, usage)
    return LLMResult(
        text=strip_think_block(result.text), tool_calls=result.tool_calls, usage=usage
    )


async def call_llm(prompt: str, config: Config | None = None) -> str:
    """Send a stateless request to the configured provider and return its answer."""
    result = await call_llm_step([user_message(prompt)], (), config)
    return result.text
