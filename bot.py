"""Telegram entry point: polling, handlers, delivery of model answers."""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from config import Config, ConfigError, describe_config, get_config
from harness import load_skills, run_agent
from llms import LLMError, LLMTimeoutError

TELEGRAM_MESSAGE_LIMIT = 4096
EMPTY_ANSWER_TEXT = "Модель вернула пустой ответ."
GENERIC_ERROR_TEXT = (
    "Не удалось получить ответ модели. Проверьте, что сервер модели запущен "
    "(ollama serve / ollama run) и что LLM_BASE_URL и LLM_MODEL заданы верно."
)
UNSUPPORTED_CONTENT_TEXT = (
    "Я понимаю только текст. Пришлите, пожалуйста, текстовое сообщение."
)
OPEN_ACCESS_WARNING = (
    "TELEGRAM_ALLOWED_IDS пуст: бот отвечает всем. У агента есть инструмент exec, "
    "поэтому для реальной работы список отправителей нужно задать."
)

logger = logging.getLogger(__name__)
router = Router()


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split text into Telegram-sized chunks by line break, then space, then hard cut."""
    if not text:
        return []
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        chunk = rest[:cut].rstrip()
        if chunk:
            parts.append(chunk)
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


async def send_answer(message: Message, text: str) -> None:
    """Deliver a model answer, splitting it and never staying silent."""
    parts = split_message(text.strip())
    if not parts:
        await message.answer(EMPTY_ANSWER_TEXT)
        return
    for part in parts:
        await message.answer(part)


def is_allowed(message: Message, config: Config) -> bool:
    """Empty TELEGRAM_ALLOWED_IDS means everyone; otherwise only the listed ids."""
    if not config.telegram_allowed_ids:
        return True
    sender = getattr(message.from_user, "id", None)
    if sender in config.telegram_allowed_ids:
        return True
    logger.warning("Сообщение от id=%s отклонено: не в TELEGRAM_ALLOWED_IDS", sender)
    return False


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    config = get_config()
    if not is_allowed(message, config):
        return
    await message.answer(
        "Привет! Я отправляю ваши сообщения языковой модели и возвращаю ответ.\n"
        f"Сейчас работает провайдер {config.llm_provider}, модель {config.llm_model}.\n"
        f"Агентный режим: до {config.agent_max_steps} шагов цикла, "
        f"инструмент exec с командами {', '.join(config.exec_allowlist)}.\n"
        "Память диалога не ведётся: каждое сообщение обрабатывается независимо."
    )


@router.message(F.text)
async def handle_text(message: Message) -> None:
    config = get_config()
    if not is_allowed(message, config):
        return
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        result = await run_agent(message.text, config)
    except LLMTimeoutError as error:
        logger.warning("LLM timeout: %s", error, exc_info=True)
        await message.answer(
            f"{error} Первый запрос после старта Ollama бывает долгим — попробуйте ещё раз."
        )
        return
    except LLMError as error:
        logger.warning("LLM error: %s", error, exc_info=True)
        await message.answer(f"{error}\n{GENERIC_ERROR_TEXT}")
        return
    except Exception:
        logger.exception("Unexpected failure while handling a message")
        await message.answer(GENERIC_ERROR_TEXT)
        return
    await send_answer(message, result.text)


@router.message()
async def handle_unsupported(message: Message) -> None:
    if not is_allowed(message, get_config()):
        return
    await message.answer(UNSUPPORTED_CONTENT_TEXT)


def build_dispatcher() -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def run_bot(config: Config) -> None:
    bot = Bot(token=config.telegram_bot_token)
    dispatcher = build_dispatcher()
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = get_config()
    except ConfigError as error:
        logger.error("Ошибка конфигурации: %s", error)
        return 1
    logger.info("Конфигурация: %s", describe_config(config))
    if not config.telegram_allowed_ids:
        logger.warning(OPEN_ACCESS_WARNING)
    try:
        load_skills(config)
    except ConfigError as error:
        logger.error("Ошибка конфигурации: %s", error)
        return 1
    try:
        asyncio.run(run_bot(config))
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    return 0


if __name__ == "__main__":
    sys.exit(main())
