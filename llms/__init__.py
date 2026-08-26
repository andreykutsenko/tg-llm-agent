"""Single entry point to the language model: call_llm(prompt) -> str."""

import re

from config import PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, Config, get_config
from llms import anthropic as anthropic_provider
from llms import ollama as ollama_provider
from llms.errors import LLMError, LLMTimeoutError

__all__ = ["call_llm", "strip_think_block", "LLMError", "LLMTimeoutError"]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Обрыв по лимиту токенов оставляет открытый <think> без закрывающего тега.
_THINK_UNCLOSED = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL | re.IGNORECASE)

_GENERATORS = {
    PROVIDER_OLLAMA: ollama_provider.generate,
    PROVIDER_ANTHROPIC: anthropic_provider.generate,
}


def strip_think_block(text: str) -> str:
    """Drop reasoning blocks of models like qwen3; keep the text if nothing remains."""
    stripped = _THINK_BLOCK.sub("", text)
    stripped = _THINK_UNCLOSED.sub("", stripped).strip()
    return stripped if stripped else text.strip()


async def call_llm(prompt: str, config: Config | None = None) -> str:
    """Send a stateless request to the configured provider and return its answer."""
    active_config = config if config is not None else get_config()
    generate = _GENERATORS.get(active_config.llm_provider)
    if generate is None:
        raise LLMError(f"Провайдер {active_config.llm_provider!r} не поддерживается.")
    return strip_think_block(await generate(prompt, active_config))
