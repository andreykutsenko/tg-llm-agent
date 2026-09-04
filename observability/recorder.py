"""Append-only JSONL of raw telemetry events; aggregation lives in bench/report.py."""

import json
import logging
import os
import time
from pathlib import Path

from config import PROJECT_ROOT
from llms.protocol import Usage

EVENT_LLM_CALL = "llm_call"
EVENT_TOOL_CALL = "tool_call"
TELEMETRY_PATH_VAR = "TELEMETRY_PATH"
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "telemetry" / "events.jsonl"

logger = logging.getLogger(__name__)

_configured_path: Path | None = None


def configure(path: Path | None) -> None:
    """Redirect events to another file (the benchmark does this per label)."""
    global _configured_path
    _configured_path = Path(path) if path is not None else None


def target_path() -> Path:
    if _configured_path is not None:
        return _configured_path
    raw = os.getenv(TELEMETRY_PATH_VAR, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_TELEMETRY_PATH


def _write(event: dict) -> None:
    """A telemetry failure is a warning, never an exception for the caller."""
    try:
        path = target_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as error:
        logger.warning("Телеметрия не записана (%s): %s", event.get("event"), error)


def record_llm_call(
    run_id: str,
    step: int,
    model: str,
    usage: Usage,
    cost: float,
    context_blocks: tuple[dict, ...] = (),
    increment: dict | None = None,
) -> None:
    """One model call. context_blocks and increment describe the payload without its content."""
    _write(
        {
            "event": EVENT_LLM_CALL,
            "ts": time.time(),
            "run_id": run_id,
            "step": step,
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_write_input_tokens": usage.cache_write_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "latency_ms": usage.latency_ms,
            "cost": cost,
            "usage_missing": not usage.is_reported,
            "context_blocks": list(context_blocks),
            "increment": increment,
        }
    )


def record_tool_call(
    run_id: str,
    step: int,
    tool_name: str,
    input_size: int,
    output_size: int,
    duration_ms: int,
) -> None:
    """One tool call: sizes and timing only, never the command or its output."""
    _write(
        {
            "event": EVENT_TOOL_CALL,
            "ts": time.time(),
            "run_id": run_id,
            "step": step,
            "tool_name": tool_name,
            "input_size": input_size,
            "output_size": output_size,
            "duration_ms": duration_ms,
        }
    )
