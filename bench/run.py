"""Benchmark runner: every task N times through the real agent, aggregate to bench/results."""

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench import report
from bench.paths import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_TELEMETRY_DIR,
    events_path,
    measurement_path,
    outcomes_path,
)
from bench.tasks import BenchTask, select_tasks
import llms
import tools
from config import Config, load_config
from harness import AgentResult, build_system_prompt, load_skills, run_agent
from observability import current_step, recorder

# Шум success rate при 5 прогонах ≥ 4 п.п. (см. REPORT-audit.md); 10 дают ≈1 п.п.
DEFAULT_RUNS = 10
DEFAULT_BENCH_MODEL = "claude-sonnet-5"
# claude-haiku-4-5 отвечает 400 на output_config.effort, поэтому по умолчанию не шлём.
DEFAULT_BENCH_EFFORT = ""
ANSWER_EXCERPT_CHARS = 300

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskOutcome:
    """One run of one task and the three-part verdict."""

    task_id: int
    slug: str
    network: bool
    run_index: int
    run_id: str
    success: bool
    check_passed: bool
    limit_reached: bool
    steps: int
    error: str
    answer_excerpt: str
    duration_ms: int


@dataclass(frozen=True)
class BenchMeta:
    """What the measurement was taken with; goes into the report verbatim."""

    label: str
    provider: str
    model: str
    effort: str | None
    runs: int
    task_ids: tuple[int, ...]
    agent_max_steps: int
    exec_max_output: int
    started_at: float
    # Константы контекста, одинаковые на каждом витке; считаются один раз.
    system_prompt_chars: int = 0
    system_prompt_tokens: int = 0
    tool_descriptions_chars: int = 0
    tool_descriptions_tokens: int = 0


def judge(task: BenchTask, result: AgentResult | None, error: str) -> tuple[bool, bool]:
    """(check_passed, success): success needs the check, no exception, no step limit."""
    if result is None:
        return (False, False)
    check_passed = task.check(result.text)
    return (check_passed, check_passed and not error and not result.limit_reached)


def bench_config(config: Config, model: str, effort: str | None) -> Config:
    """The measurement pins the model and effort; everything else is the bot's config."""
    return dataclasses.replace(config, llm_model=model, llm_effort=effort, llm_count_tokens=True)


PROBE_MESSAGE = "."


async def measure_constants(config: Config) -> dict:
    """Sizes of the system prompt (with skills) and tool descriptions, once per measurement."""
    system_prompt = build_system_prompt(config, load_skills(config))
    specs = tools.tool_specs()
    tool_descriptions = json.dumps([dataclasses.asdict(spec) for spec in specs], ensure_ascii=False)
    probe = [llms.user_message(PROBE_MESSAGE)]
    with_system = await llms.count_tokens(probe, (), config, system=system_prompt)
    with_tools = await llms.count_tokens(probe, specs, config, system=system_prompt)
    probe_only = await llms.count_tokens(probe, (), config)
    countable = None not in (with_system, with_tools, probe_only)
    return {
        "system_prompt_chars": len(system_prompt),
        "system_prompt_tokens": with_system - probe_only if countable else 0,
        "tool_descriptions_chars": len(tool_descriptions),
        "tool_descriptions_tokens": with_tools - with_system if countable else 0,
    }


def _run_id_after(previous) -> str:
    position = current_step()
    if position is None or position is previous:
        return ""
    return position.run_id


async def run_task_once(task: BenchTask, config: Config, run_index: int) -> TaskOutcome:
    """A failing task is an outcome with an error text, never an exception."""
    previous = current_step()
    started = time.perf_counter()
    result: AgentResult | None = None
    error = ""
    try:
        result = await run_agent(task.prompt, config)
    except Exception as failure:  # noqa: BLE001 — любая ошибка задачи идёт в отчёт
        error = f"{type(failure).__name__}: {failure}"
        logger.warning("Задача %d (%s) упала: %s", task.id, task.slug, error)
    duration_ms = int((time.perf_counter() - started) * 1000)
    check_passed, success = judge(task, result, error)
    return TaskOutcome(
        task_id=task.id,
        slug=task.slug,
        network=task.network,
        run_index=run_index,
        run_id=_run_id_after(previous),
        success=success,
        check_passed=check_passed,
        limit_reached=result.limit_reached if result else False,
        steps=result.steps if result else 0,
        error=error,
        answer_excerpt=(result.text if result else "")[:ANSWER_EXCERPT_CHARS],
        duration_ms=duration_ms,
    )


async def run_benchmark(
    tasks: tuple[BenchTask, ...],
    runs: int,
    config: Config,
    on_outcome: Callable[[TaskOutcome, int, int], None] | None = None,
) -> list[TaskOutcome]:
    """Sequential runs: the local model and the API rate limit both prefer that."""
    outcomes: list[TaskOutcome] = []
    total = len(tasks) * runs
    for run_index in range(1, runs + 1):
        for task in tasks:
            outcome = await run_task_once(task, config, run_index)
            outcomes.append(outcome)
            if on_outcome is not None:
                on_outcome(outcome, len(outcomes), total)
    return outcomes


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_measurement(meta: BenchMeta, outcomes: list[TaskOutcome], events: list[dict]) -> dict:
    """Everything the report needs, without command contents: meta, summary, one timeline."""
    outcome_records = [dataclasses.asdict(outcome) for outcome in outcomes]
    summary = report.aggregate(dataclasses.asdict(meta), outcome_records, events)
    timeline = report.choose_timeline_run(events)
    return {
        "meta": dataclasses.asdict(meta),
        "summary": dataclasses.asdict(summary),
        "timeline": dataclasses.asdict(timeline),
    }


def write_measurement(path: Path, measurement: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(measurement, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def success_rate(outcomes: list[TaskOutcome], network: bool) -> float | None:
    subset = [outcome for outcome in outcomes if outcome.network == network]
    if not subset:
        return None
    return sum(1 for outcome in subset if outcome.success) / len(subset)


def _print_progress(outcome: TaskOutcome, done: int, total: int) -> None:
    verdict = "OK " if outcome.success else "FAIL"
    detail = outcome.error or outcome.answer_excerpt.replace("\n", " ")[:80]
    print(
        f"[{done}/{total}] {verdict} задача {outcome.task_id:2d} {outcome.slug:<20} "
        f"прогон {outcome.run_index} шагов={outcome.steps} "
        f"{outcome.duration_ms / 1000:.1f}с | {detail}",
        flush=True,
    )


def _parse_task_ids(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(item) for item in raw.split(",") if item.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Прогон бенчмарка агента.")
    parser.add_argument("--label", required=True, help="имя замера: before, after-cache, ...")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="прогонов на задачу")
    parser.add_argument("--model", default=DEFAULT_BENCH_MODEL, help="модель замера")
    parser.add_argument(
        "--effort", default=DEFAULT_BENCH_EFFORT, help="output_config.effort; пусто — не отправлять"
    )
    parser.add_argument("--tasks", default="", help="id задач через запятую; пусто — все")
    parser.add_argument("--no-network", action="store_true", help="пропустить сетевые задачи")
    parser.add_argument("--dir", type=Path, default=DEFAULT_RESULTS_DIR, help="каталог агрегатов (в git)")
    parser.add_argument(
        "--telemetry-dir", type=Path, default=DEFAULT_TELEMETRY_DIR, help="каталог сырого JSONL"
    )
    parser.add_argument("--force", action="store_true", help="перезаписать существующий замер")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    measurement_file = measurement_path(args.dir, args.label)
    if measurement_file.exists() and not args.force:
        print(f"Замер {args.label!r} уже есть: {measurement_file}; выберите другой --label "
              "или укажите --force.", file=sys.stderr)
        return 2
    config = bench_config(load_config(), args.model, args.effort or None)
    tasks = select_tasks(_parse_task_ids(args.tasks), include_network=not args.no_network)
    constants = asyncio.run(measure_constants(config))
    meta = BenchMeta(
        label=args.label,
        provider=config.llm_provider,
        model=config.llm_model,
        effort=config.llm_effort,
        runs=args.runs,
        task_ids=tuple(task.id for task in tasks),
        agent_max_steps=config.agent_max_steps,
        exec_max_output=config.exec_max_output,
        started_at=time.time(),
        **constants,
    )
    events_file = events_path(args.telemetry_dir, args.label)
    events_file.parent.mkdir(parents=True, exist_ok=True)
    events_file.write_text("", encoding="utf-8")
    recorder.configure(events_file)
    print(f"Замер {args.label}: модель {meta.model} ({meta.provider}), effort={meta.effort}, "
          f"задач {len(tasks)}, прогонов {args.runs}; system prompt {meta.system_prompt_tokens} tok, "
          f"описания инструментов {meta.tool_descriptions_tokens} tok")
    outcomes = asyncio.run(run_benchmark(tasks, args.runs, config, _print_progress))
    write_jsonl(outcomes_path(args.telemetry_dir, args.label), [dataclasses.asdict(o) for o in outcomes])
    measurement = build_measurement(meta, outcomes, report.read_jsonl(events_file))
    write_measurement(measurement_file, measurement)
    local = success_rate(outcomes, network=False)
    network = success_rate(outcomes, network=True)
    by_run = measurement["summary"]["success_rate_by_run"]
    spread = f" (по прогонам: {', '.join(f'{r:.0%}' for r in by_run)})" if len(by_run) > 1 else ""
    print(f"Локальные задачи: success rate {local:.0%}{spread}" if local is not None else "Локальных задач нет")
    if network is not None:
        print(f"Сетевые задачи (flaky): success rate {network:.0%}")
    print(f"Агрегат: {measurement_file}\nСырьё:   {events_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
