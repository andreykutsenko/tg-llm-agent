"""Cloud model behind the native Anthropic SDK."""

from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic

from config import Config
from llms.errors import LLMError, LLMTimeoutError


def extract_text(content_blocks) -> str:
    """Concatenate text blocks of a Messages API response."""
    return "".join(
        block.text for block in content_blocks if getattr(block, "type", None) == "text"
    )


async def generate(prompt: str, config: Config) -> str:
    client = AsyncAnthropic(
        api_key=config.llm_api_key,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    try:
        response = await client.messages.create(
            model=config.llm_model,
            system=config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.llm_max_tokens,
        )
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

    return extract_text(response.content)
