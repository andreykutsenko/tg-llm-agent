"""Tool registry: descriptions for the model plus the functions that run them."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

from config import Config
from llms.protocol import ToolSpec
from observability import current_step
from observability import recorder
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


def _argument_size(arguments: dict) -> int:
    try:
        return len(json.dumps(arguments, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(arguments))


def _record(name: str, arguments: dict, output: str, started: float) -> None:
    position = current_step()
    if position is None:
        return
    recorder.record_tool_call(
        run_id=position.run_id,
        step=position.step,
        tool_name=name,
        input_size=_argument_size(arguments),
        output_size=len(output),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


async def _dispatch(name: str, arguments: dict, config: Config) -> str:
    tool = _BY_NAME.get(name)
    if tool is None:
        known = ", ".join(_BY_NAME) or "нет доступных инструментов"
        logger.info("Модель запросила неизвестный инструмент %r", name)
        return f"Инструмент {name!r} не существует. Доступны: {known}."
    return await asyncio.to_thread(tool.run, arguments, config)


async def run_tool(name: str, arguments: dict, config: Config) -> str:
    """Run a requested tool; an unknown name is a normal answer, not a failure."""
    started = time.perf_counter()
    output = await _dispatch(name, arguments, config)
    _record(name, arguments, output, started)
    return output
