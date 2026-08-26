"""Local model behind an OpenAI-compatible /v1 API (Ollama, vLLM)."""

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from config import Config
from llms.errors import LLMError, LLMTimeoutError


def build_messages(prompt: str, config: Config) -> list[dict[str, str]]:
    """System prompt plus the current user message — no dialog history."""
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": prompt},
    ]


async def generate(prompt: str, config: Config) -> str:
    client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    try:
        response = await client.chat.completions.create(
            model=config.llm_model,
            messages=build_messages(prompt, config),
            max_tokens=config.llm_max_tokens,
        )
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

    choices = response.choices
    if not choices:
        return ""
    return choices[0].message.content or ""
