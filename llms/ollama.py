"""Local model behind an OpenAI-compatible /v1 API (Ollama, vLLM)."""

import json

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from config import Config
from llms.errors import LLMError, LLMTimeoutError
from llms.protocol import (
    INVALID_ARGUMENTS_KEY,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    LLMResult,
    ToolCall,
    ToolSpec,
    user_message,
)


def describe_tool(tool: ToolSpec) -> dict:
    """OpenAI-compatible tool description."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _assistant_payload(message: dict) -> dict:
    payload = {"role": ROLE_ASSISTANT, "content": message.get("content") or ""}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in tool_calls
        ]
    return payload


def build_messages(messages: list[dict], config: Config) -> list[dict]:
    """System prompt plus the messages of the current request — no dialog history."""
    payload: list[dict] = [{"role": "system", "content": config.system_prompt}]
    for message in messages:
        role = message["role"]
        if role == ROLE_ASSISTANT:
            payload.append(_assistant_payload(message))
        elif role == ROLE_TOOL:
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )
        else:
            payload.append({"role": "user", "content": message["content"]})
    return payload


def _parse_arguments(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        arguments = json.loads(raw)
    except (TypeError, ValueError):
        return {INVALID_ARGUMENTS_KEY: raw}
    if not isinstance(arguments, dict):
        return {INVALID_ARGUMENTS_KEY: raw}
    return arguments


def parse_response(response) -> LLMResult:
    """Reduce an OpenAI-compatible answer to the provider-independent result."""
    choices = response.choices
    if not choices:
        return LLMResult()
    message = choices[0].message
    raw_calls = getattr(message, "tool_calls", None) or []
    calls = tuple(
        ToolCall(
            id=getattr(call, "id", None) or f"call_{index}",
            name=call.function.name,
            arguments=_parse_arguments(call.function.arguments),
        )
        for index, call in enumerate(raw_calls)
    )
    return LLMResult(text=message.content or "", tool_calls=calls)


async def generate_step(
    messages: list[dict], tools: tuple[ToolSpec, ...], config: Config
) -> LLMResult:
    """One model call: either a final text or requested tool calls."""
    client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    request = {
        "model": config.llm_model,
        "messages": build_messages(messages, config),
        "max_tokens": config.llm_max_tokens,
    }
    if tools:
        request["tools"] = [describe_tool(tool) for tool in tools]
    try:
        response = await client.chat.completions.create(**request)
    except APITimeoutError as error:
        raise LLMTimeoutError(
            f"Модель {config.llm_model} не ответила за {config.llm_timeout_seconds:.0f} секунд."
        ) from error
    except APIConnectionError as error:
        raise LLMError(
            f"Не удалось подключиться к {config.llm_base_url}."
        ) from error
    except APIError as error:
        raise LLMError(f"Сервер модели вернул ошибку: {error}") from error
    finally:
        await client.close()

    return parse_response(response)


async def generate(prompt: str, config: Config) -> str:
    result = await generate_step([user_message(prompt)], (), config)
    return result.text
