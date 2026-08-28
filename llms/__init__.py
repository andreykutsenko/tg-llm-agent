"""Single entry point to the language model: text answers and tool-call steps."""

import re

from config import PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, Config, get_config
from llms import anthropic as anthropic_provider
from llms import ollama as ollama_provider
from llms.errors import LLMError, LLMTimeoutError
from llms.protocol import (
    LLMResult,
    ToolCall,
    ToolSpec,
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


async def call_llm_step(
    messages: list[dict],
    tools: tuple[ToolSpec, ...] = (),
    config: Config | None = None,
) -> LLMResult:
    """One step of the agent loop, in the same shape for every provider."""
    active_config, generate_step = _resolve(config)
    result = await generate_step(messages, tools, active_config)
    return LLMResult(text=strip_think_block(result.text), tool_calls=result.tool_calls)


async def call_llm(prompt: str, config: Config | None = None) -> str:
    """Send a stateless request to the configured provider and return its answer."""
    result = await call_llm_step([user_message(prompt)], (), config)
    return result.text
