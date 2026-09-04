"""Cloud model behind the native Anthropic SDK."""

from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic

from config import Config
from llms.errors import LLMError, LLMTimeoutError
from llms.protocol import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    LLMResult,
    ToolCall,
    ToolSpec,
    Usage,
    user_message,
)


def describe_tool(tool: ToolSpec) -> dict:
    """Native Anthropic tool description."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def extract_text(content_blocks) -> str:
    """Concatenate text blocks of a Messages API response."""
    return "".join(
        block.text for block in content_blocks if getattr(block, "type", None) == "text"
    )


def _assistant_content(message: dict) -> list[dict]:
    blocks: list[dict] = []
    text = message.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return blocks


def build_messages(messages: list[dict]) -> list[dict]:
    """Tool results are user-side blocks here, so consecutive ones are merged."""
    payload: list[dict] = []
    pending_results: list[dict] = []

    def flush_results() -> None:
        if pending_results:
            payload.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message["role"]
        if role == ROLE_TOOL:
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )
            continue
        flush_results()
        if role == ROLE_ASSISTANT:
            payload.append({"role": "assistant", "content": _assistant_content(message)})
        else:
            payload.append({"role": "user", "content": message["content"]})
    flush_results()
    return payload


def _count(usage, field: str) -> int:
    return int(getattr(usage, field, None) or 0)


def parse_usage(response) -> Usage:
    """Token counts of a Messages API response; a missing block gives zeros."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    details = getattr(usage, "output_tokens_details", None)
    return Usage(
        input_tokens=_count(usage, "input_tokens"),
        output_tokens=_count(usage, "output_tokens"),
        cached_input_tokens=_count(usage, "cache_read_input_tokens"),
        cache_write_input_tokens=_count(usage, "cache_creation_input_tokens"),
        reasoning_tokens=_count(details, "thinking_tokens") if details is not None else 0,
    )


def parse_response(response) -> LLMResult:
    """Reduce a native Anthropic answer to the provider-independent result."""
    blocks = response.content
    calls = tuple(
        ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
        for block in blocks
        if getattr(block, "type", None) == "tool_use"
    )
    return LLMResult(text=extract_text(blocks), tool_calls=calls, usage=parse_usage(response))


async def generate_step(
    messages: list[dict], tools: tuple[ToolSpec, ...], config: Config
) -> LLMResult:
    """One model call: either a final text or requested tool calls."""
    client = AsyncAnthropic(
        api_key=config.llm_api_key,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    request = {
        "model": config.llm_model,
        "system": config.system_prompt,
        "messages": build_messages(messages),
        "max_tokens": config.llm_max_tokens,
    }
    if config.llm_effort is not None:
        request["output_config"] = {"effort": config.llm_effort}
    if tools:
        request["tools"] = [describe_tool(tool) for tool in tools]
    try:
        response = await client.messages.create(**request)
    except APITimeoutError as error:
        raise LLMTimeoutError(
            f"Модель {config.llm_model} не ответила за {config.llm_timeout_seconds:.0f} секунд."
        ) from error
    except APIConnectionError as error:
        raise LLMError("Не удалось подключиться к API Anthropic.") from error
    except APIError as error:
        raise LLMError(f"API Anthropic вернул ошибку: {error}") from error
    finally:
        await client.close()

    return parse_response(response)


async def count_tokens(
    messages: list[dict],
    tools: tuple[ToolSpec, ...],
    config: Config,
    system: str | None = None,
) -> int:
    """Free token count of a payload fragment with the model's own tokenizer."""
    client = AsyncAnthropic(
        api_key=config.llm_api_key, timeout=config.llm_timeout_seconds, max_retries=0
    )
    request = {"model": config.llm_model, "messages": build_messages(messages)}
    if system:
        request["system"] = system
    if tools:
        request["tools"] = [describe_tool(tool) for tool in tools]
    try:
        response = await client.messages.count_tokens(**request)
    except (APITimeoutError, APIConnectionError, APIError) as error:
        raise LLMError(f"count_tokens не удался: {error}") from error
    finally:
        await client.close()
    return int(response.input_tokens)


async def generate(prompt: str, config: Config) -> str:
    result = await generate_step([user_message(prompt)], (), config)
    return result.text
