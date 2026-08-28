"""Tool registry: descriptions for the model plus the functions that run them."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from config import Config
from llms.protocol import ToolSpec
from tools import exec as exec_tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    """A tool as the model sees it and as the harness runs it."""

    spec: ToolSpec
    run: Callable[[dict, Config], str]


REGISTRY: tuple[Tool, ...] = (Tool(spec=exec_tool.tool_spec(), run=exec_tool.run_exec),)

_BY_NAME = {tool.spec.name: tool for tool in REGISTRY}


def tool_specs() -> tuple[ToolSpec, ...]:
    """Descriptions passed to the model on every step."""
    return tuple(tool.spec for tool in REGISTRY)


def describe_tools() -> str:
    """Short listing of the tools for the system prompt."""
    return "\n".join(f"- {tool.spec.name}: {tool.spec.description}" for tool in REGISTRY)


async def run_tool(name: str, arguments: dict, config: Config) -> str:
    """Run a requested tool; an unknown name is a normal answer, not a failure."""
    tool = _BY_NAME.get(name)
    if tool is None:
        known = ", ".join(_BY_NAME) or "нет доступных инструментов"
        logger.info("Модель запросила неизвестный инструмент %r", name)
        return f"Инструмент {name!r} не существует. Доступны: {known}."
    return await asyncio.to_thread(tool.run, arguments, config)
