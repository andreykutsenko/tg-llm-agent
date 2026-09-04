"""Aggregation of raw JSONL into the audit numbers, a before/after comparison, a dashboard."""

import argparse
import html
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bench.paths import DEFAULT_RESULTS_DIR, measurement_path
from bench.pricing import cost_breakdown
from llms.protocol import Usage
from observability.recorder import EVENT_LLM_CALL, EVENT_TOOL_CALL

KIND_SYSTEM = "system"
KIND_TOOLS = "tools"
KIND_USER = "user"
KIND_ASSISTANT = "assistant"
KIND_TOOL = "tool"
CONTEXT_KINDS = (KIND_SYSTEM, KIND_TOOLS, KIND_USER, KIND_ASSISTANT, KIND_TOOL)
KIND_TITLES = {
    KIND_SYSTEM: "system prompt",
    KIND_TOOLS: "описания инструментов",
    KIND_USER: "сообщение пользователя",
    KIND_ASSISTANT: "ответы модели (история)",
    KIND_TOOL: "вывод инструментов",
}
DEFAULT_DASHBOARD_NAME = "dashboard.html"


# ── loading ────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── per-step arithmetic ────────────────────────────────────────────────────


def event_usage(event: dict) -> Usage:
    return Usage(
        input_tokens=int(event.get("input_tokens", 0)),
        output_tokens=int(event.get("output_tokens", 0)),
        cached_input_tokens=int(event.get("cached_input_tokens", 0)),
        cache_write_input_tokens=int(event.get("cache_write_input_tokens", 0)),
        reasoning_tokens=int(event.get("reasoning_tokens", 0)),
        latency_ms=int(event.get("latency_ms", 0)),
    )


def context_tokens(event: dict) -> int:
    """Everything the model read on this step, cached or not."""
    usage = event_usage(event)
    return usage.input_tokens + usage.cached_input_tokens + usage.cache_write_input_tokens


def block_token_estimates(event: dict) -> list[tuple[dict, float]]:
    """Split the step's input tokens across payload blocks proportionally to their size."""
    blocks = event.get("context_blocks") or []
    total_chars = sum(int(block.get("chars", 0)) for block in blocks)
    total = context_tokens(event)
    if not blocks or total_chars == 0:
        return [(block, 0.0) for block in blocks]
    return [(block, total * int(block.get("chars", 0)) / total_chars) for block in blocks]


def tokens_by_kind(event: dict) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for block, tokens in block_token_estimates(event):
        result[block.get("kind", KIND_USER)] += tokens
    return dict(result)


@dataclass(frozen=True)
class StepView:
    step: int
    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int
    context_tokens: int
    repeated_tokens: float
    cost: float
    latency_ms: int
    by_kind: dict[str, float]


@dataclass(frozen=True)
class RunTimeline:
    run_id: str
    steps: tuple[StepView, ...]

    @property
    def total_cost(self) -> float:
        return sum(step.cost for step in self.steps)


def llm_events_of_run(events: list[dict], run_id: str) -> list[dict]:
    return sorted(
        (e for e in events if e.get("event") == EVENT_LLM_CALL and e.get("run_id") == run_id),
        key=lambda e: int(e.get("step", 0)),
    )


def repeated_tokens_per_step(llm_events: list[dict]) -> list[float]:
    """Tokens of blocks that were already in the previous step's payload of the same run."""
    repeated: list[float] = []
    previous_hashes: set[str] = set()
    for event in llm_events:
        estimates = block_token_estimates(event)
        repeated.append(sum(t for block, t in estimates if block.get("hash") in previous_hashes))
        previous_hashes = {block.get("hash") for block, _ in estimates}
    return repeated


def count_repeated_tokens(llm_events: list[dict]) -> tuple[float, int]:
    """(repeated tokens, total input tokens) over the whole run."""
    return (
        sum(repeated_tokens_per_step(llm_events)),
        sum(context_tokens(event) for event in llm_events),
    )


def build_timeline(events: list[dict], run_id: str) -> RunTimeline:
    llm_events = llm_events_of_run(events, run_id)
    repeated = repeated_tokens_per_step(llm_events)
    steps = tuple(
        StepView(
            step=int(event.get("step", index + 1)),
            input_tokens=int(event.get("input_tokens", 0)),
            cached_input_tokens=int(event.get("cached_input_tokens", 0)),
            cache_write_input_tokens=int(event.get("cache_write_input_tokens", 0)),
            output_tokens=int(event.get("output_tokens", 0)),
            context_tokens=context_tokens(event),
            repeated_tokens=repeated[index],
            cost=float(event.get("cost", 0.0)),
            latency_ms=int(event.get("latency_ms", 0)),
            by_kind=tokens_by_kind(event),
        )
        for index, event in enumerate(llm_events)
    )
    return RunTimeline(run_id=run_id, steps=steps)


# ── aggregation ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolStat:
    name: str
    calls: int
    output_chars: int
    estimated_tokens: float


@dataclass(frozen=True)
class TaskStat:
    task_id: int
    slug: str
    network: bool
    runs: int
    successes: int
    avg_cost: float
    avg_steps: float
    avg_context_tokens: float


@dataclass(frozen=True)
class Summary:
    label: str
    provider: str
    model: str
    temperature: float | None
    runs: int
    task_count: int
    outcome_count: int
    success_rate_local: float | None
    success_rate_network: float | None
    tokens: dict[str, int]
    cost: dict[str, float]
    avg_cost_per_run: float
    avg_tokens_per_run: float
    cache_hit_rate: float
    tools: tuple[ToolStat, ...]
    repeated_tokens: float
    repeated_share: float
    most_expensive_step: tuple[str, int, float] | None
    growth_per_step: dict[str, float]
    per_task: tuple[TaskStat, ...]
    llm_calls: int = 0
    usage_missing_calls: int = 0
    task_runs: dict[int, tuple[str, ...]] = field(default_factory=dict)


def _success_rate(outcomes: list[dict], network: bool) -> float | None:
    subset = [o for o in outcomes if bool(o.get("network")) == network]
    if not subset:
        return None
    return sum(1 for o in subset if o.get("success")) / len(subset)


def _sum_tokens(llm_events: list[dict]) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "cached": 0, "cache_write": 0, "reasoning": 0}
    for event in llm_events:
        usage = event_usage(event)
        totals["input"] += usage.input_tokens
        totals["output"] += usage.output_tokens
        totals["cached"] += usage.cached_input_tokens
        totals["cache_write"] += usage.cache_write_input_tokens
        totals["reasoning"] += usage.reasoning_tokens
    return totals


def _sum_cost(llm_events: list[dict]) -> dict[str, float]:
    totals = {"input": 0.0, "output": 0.0, "cached": 0.0, "cache_write": 0.0}
    for event in llm_events:
        breakdown = cost_breakdown(str(event.get("model", "")), event_usage(event))
        totals["input"] += breakdown.input
        totals["output"] += breakdown.output
        totals["cached"] += breakdown.cached_input
        totals["cache_write"] += breakdown.cache_write
    totals["total"] = sum(totals.values())
    return totals


def _tool_stats(events: list[dict]) -> tuple[ToolStat, ...]:
    """Tool output tokens: the fresh tool blocks of step s belong to the tool calls of step s-1."""
    calls: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    tokens: dict[str, float] = defaultdict(float)
    tool_events: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for event in events:
        if event.get("event") == EVENT_TOOL_CALL:
            name = str(event.get("tool_name", "?"))
            calls[name] += 1
            chars[name] += int(event.get("output_size", 0))
            tool_events[(str(event.get("run_id")), int(event.get("step", 0)))].append(event)
    for event in events:
        if event.get("event") != EVENT_LLM_CALL:
            continue
        producers = tool_events.get((str(event.get("run_id")), int(event.get("step", 0)) - 1), [])
        if not producers:
            continue
        tool_blocks = [t for block, t in block_token_estimates(event) if block.get("kind") == KIND_TOOL]
        fresh = tool_blocks[-len(producers):]
        for producer, block_tokens in zip(producers, fresh):
            tokens[str(producer.get("tool_name", "?"))] += block_tokens
    return tuple(
        sorted(
            (ToolStat(name, calls[name], chars[name], tokens[name]) for name in calls),
            key=lambda stat: stat.estimated_tokens,
            reverse=True,
        )
    )


def _growth_per_step(timelines: list[RunTimeline]) -> dict[str, float]:
    """Average tokens a context kind gains per step, over runs with at least two steps."""
    growth: dict[str, list[float]] = defaultdict(list)
    for timeline in timelines:
        if len(timeline.steps) < 2:
            continue
        first, last = timeline.steps[0], timeline.steps[-1]
        span = len(timeline.steps) - 1
        for kind in CONTEXT_KINDS:
            growth[kind].append((last.by_kind.get(kind, 0.0) - first.by_kind.get(kind, 0.0)) / span)
    return {kind: (sum(values) / len(values) if values else 0.0) for kind, values in growth.items()}


def _per_task(outcomes: list[dict], timelines_by_run: dict[str, RunTimeline]) -> tuple[TaskStat, ...]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for outcome in outcomes:
        grouped[int(outcome["task_id"])].append(outcome)
    stats = []
    for task_id, items in sorted(grouped.items()):
        timelines = [timelines_by_run[o["run_id"]] for o in items if o.get("run_id") in timelines_by_run]
        costs = [t.total_cost for t in timelines]
        contexts = [max((s.context_tokens for s in t.steps), default=0) for t in timelines]
        stats.append(
            TaskStat(
                task_id=task_id,
                slug=str(items[0].get("slug", "")),
                network=bool(items[0].get("network")),
                runs=len(items),
                successes=sum(1 for o in items if o.get("success")),
                avg_cost=sum(costs) / len(costs) if costs else 0.0,
                avg_steps=sum(int(o.get("steps", 0)) for o in items) / len(items),
                avg_context_tokens=sum(contexts) / len(contexts) if contexts else 0.0,
            )
        )
    return tuple(stats)


def aggregate(meta: dict, outcomes: list[dict], events: list[dict]) -> Summary:
    llm_events = [e for e in events if e.get("event") == EVENT_LLM_CALL]
    run_ids = sorted({str(e.get("run_id")) for e in llm_events})
    timelines = [build_timeline(events, run_id) for run_id in run_ids]
    timelines_by_run = {t.run_id: t for t in timelines}
    tokens = _sum_tokens(llm_events)
    cost = _sum_cost(llm_events)
    repeated = sum(count_repeated_tokens(llm_events_of_run(events, r))[0] for r in run_ids)
    total_input = tokens["input"] + tokens["cached"] + tokens["cache_write"]
    outcome_count = len(outcomes)
    most_expensive = max(
        ((t.run_id, s.step, s.cost) for t in timelines for s in t.steps),
        key=lambda item: item[2],
        default=None,
    )
    task_runs: dict[int, list[str]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.get("run_id"):
            task_runs[int(outcome["task_id"])].append(str(outcome["run_id"]))
    return Summary(
        label=str(meta.get("label", "")),
        provider=str(meta.get("provider", "")),
        model=str(meta.get("model", "")),
        temperature=meta.get("temperature"),
        runs=int(meta.get("runs", 0)),
        task_count=len({o.get("task_id") for o in outcomes}),
        outcome_count=outcome_count,
        success_rate_local=_success_rate(outcomes, network=False),
        success_rate_network=_success_rate(outcomes, network=True),
        tokens=tokens,
        cost=cost,
        avg_cost_per_run=cost["total"] / outcome_count if outcome_count else 0.0,
        avg_tokens_per_run=(total_input + tokens["output"]) / outcome_count if outcome_count else 0.0,
        cache_hit_rate=tokens["cached"] / total_input if total_input else 0.0,
        tools=_tool_stats(events),
        repeated_tokens=repeated,
        repeated_share=repeated / total_input if total_input else 0.0,
        most_expensive_step=most_expensive,
        growth_per_step=_growth_per_step(timelines),
        per_task=_per_task(outcomes, timelines_by_run),
        llm_calls=len(llm_events),
        usage_missing_calls=sum(1 for e in llm_events if e.get("usage_missing")),
        task_runs={task_id: tuple(ids) for task_id, ids in task_runs.items()},
    )


# ── persisted measurement ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """What bench.run writes to bench/results/<label>.json."""

    meta: dict
    summary: Summary
    timeline: RunTimeline


def summary_from_dict(data: dict) -> Summary:
    fields = dict(data)
    fields["tools"] = tuple(ToolStat(**item) for item in fields.get("tools", ()))
    fields["per_task"] = tuple(TaskStat(**item) for item in fields.get("per_task", ()))
    expensive = fields.get("most_expensive_step")
    fields["most_expensive_step"] = tuple(expensive) if expensive else None
    fields["task_runs"] = {int(k): tuple(v) for k, v in (fields.get("task_runs") or {}).items()}
    return Summary(**fields)


def timeline_from_dict(data: dict) -> RunTimeline:
    return RunTimeline(
        run_id=str(data.get("run_id", "")),
        steps=tuple(StepView(**step) for step in data.get("steps", ())),
    )


def load_measurement(directory: Path, label: str) -> Measurement:
    data = json.loads(measurement_path(directory, label).read_text(encoding="utf-8"))
    return Measurement(
        meta=data.get("meta", {}),
        summary=summary_from_dict(data["summary"]),
        timeline=timeline_from_dict(data.get("timeline", {})),
    )


# ── comparison ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Delta:
    metric: str
    before: float
    after: float
    delta_pct: float | None
    unit: str = ""

    @property
    def delta_points(self) -> float:
        return self.after - self.before


def percent_delta(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before * 100.0


def compare(before: Summary, after: Summary) -> tuple[Delta, ...]:
    """Cost and tokens as percentages, success rate as percentage points."""
    pairs = [
        ("стоимость, всего", before.cost["total"], after.cost["total"], "$"),
        ("стоимость input", before.cost["input"], after.cost["input"], "$"),
        ("стоимость output", before.cost["output"], after.cost["output"], "$"),
        ("стоимость cached input", before.cost["cached"], after.cost["cached"], "$"),
        ("стоимость cache write", before.cost["cache_write"], after.cost["cache_write"], "$"),
        ("средняя стоимость задачи", before.avg_cost_per_run, after.avg_cost_per_run, "$"),
        ("input tokens", before.tokens["input"], after.tokens["input"], "tok"),
        ("cached input tokens", before.tokens["cached"], after.tokens["cached"], "tok"),
        ("output tokens", before.tokens["output"], after.tokens["output"], "tok"),
        ("повторно отправленные tokens", before.repeated_tokens, after.repeated_tokens, "tok"),
        ("cache hit rate", before.cache_hit_rate * 100, after.cache_hit_rate * 100, "%"),
        (
            "success rate (локальные)",
            (before.success_rate_local or 0.0) * 100,
            (after.success_rate_local or 0.0) * 100,
            "п.п.",
        ),
    ]
    return tuple(
        Delta(
            metric=name,
            before=b,
            after=a,
            delta_pct=None if unit == "п.п." else percent_delta(b, a),
            unit=unit,
        )
        for name, b, a, unit in pairs
    )


# ── dashboard ──────────────────────────────────────────────────────────────

_CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fafafa}
h1,h2{margin:24px 0 8px}table{border-collapse:collapse;margin:8px 0 16px;background:#fff}
th,td{border:1px solid #ddd;padding:4px 10px;text-align:right}th{background:#f0f0f0}
td:first-child,th:first-child{text-align:left}.muted{color:#666}.card{display:inline-block;
border:1px solid #ddd;background:#fff;padding:8px 14px;margin:4px 8px 4px 0;min-width:140px}
.card b{display:block;font-size:20px}svg{background:#fff;border:1px solid #ddd}
.legend span{display:inline-block;margin-right:14px}.legend i{display:inline-block;width:12px;
height:12px;margin-right:4px;vertical-align:-1px}
"""
_COLOR_INPUT = "#4a78c2"
_COLOR_CACHED = "#9cc0f0"
_COLOR_WRITE = "#f0b64a"
_COLOR_OUTPUT = "#d9534f"
_COLOR_CONTEXT = "#2a9d5c"


def _fmt_money(value: float) -> str:
    return f"${value:.4f}"


def _fmt_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _card(title: str, value: str) -> str:
    return f'<div class="card"><span class="muted">{html.escape(title)}</span><b>{html.escape(value)}</b></div>'


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _summary_section(summary: Summary) -> str:
    cards = "".join(
        [
            _card("замер", summary.label),
            _card("модель", summary.model),
            _card("temperature", "—" if summary.temperature is None else str(summary.temperature)),
            _card("задач × прогонов", f"{summary.task_count} × {summary.runs}"),
            _card("success rate (локальные)", _fmt_rate(summary.success_rate_local)),
            _card("success rate (сетевые, flaky)", _fmt_rate(summary.success_rate_network)),
            _card("стоимость", _fmt_money(summary.cost["total"])),
            _card("в среднем на задачу", _fmt_money(summary.avg_cost_per_run)),
            _card("cache hit rate", _fmt_rate(summary.cache_hit_rate)),
            _card("повторно отправлено", f"{_fmt_int(summary.repeated_tokens)} tok ({summary.repeated_share * 100:.0f}%)"),
        ]
    )
    tokens = _table(
        ("тип", "tokens", "стоимость"),
        [
            ("input (без кэша)", _fmt_int(summary.tokens["input"]), _fmt_money(summary.cost["input"])),
            ("cached input", _fmt_int(summary.tokens["cached"]), _fmt_money(summary.cost["cached"])),
            ("cache write", _fmt_int(summary.tokens["cache_write"]), _fmt_money(summary.cost["cache_write"])),
            ("output", _fmt_int(summary.tokens["output"]), _fmt_money(summary.cost["output"])),
            ("  из них reasoning", _fmt_int(summary.tokens.get("reasoning", 0)), "входит в output"),
            ("итого", "", _fmt_money(summary.cost["total"])),
        ],
    )
    tools = _table(
        ("инструмент", "вызовов", "символов вывода", "≈ tokens в контексте"),
        [(t.name, str(t.calls), _fmt_int(t.output_chars), _fmt_int(t.estimated_tokens)) for t in summary.tools]
        or [("—", "0", "0", "0")],
    )
    growth = _table(
        ("тип контекста", "≈ tokens прироста за виток"),
        [(KIND_TITLES.get(k, k), _fmt_int(v)) for k, v in sorted(summary.growth_per_step.items(), key=lambda kv: -kv[1])],
    )
    expensive = "—"
    if summary.most_expensive_step:
        run_id, step, cost = summary.most_expensive_step
        expensive = f"run {run_id[:8]}, виток {step}: {_fmt_money(cost)}"
    tasks = _table(
        ("задача", "успехов", "ср. стоимость", "ср. шагов", "ср. макс. контекст"),
        [
            (
                f"{t.task_id}. {t.slug}{' (сеть)' if t.network else ''}",
                f"{t.successes}/{t.runs}",
                _fmt_money(t.avg_cost),
                f"{t.avg_steps:.1f}",
                _fmt_int(t.avg_context_tokens),
            )
            for t in summary.per_task
        ],
    )
    missing = (
        f'<p class="muted">Вызовов без usage от провайдера: {summary.usage_missing_calls} из {summary.llm_calls}.</p>'
        if summary.usage_missing_calls
        else ""
    )
    return (
        f"<h2>Сводка: {html.escape(summary.label)}</h2>{cards}{missing}"
        f"<h3>Токены и стоимость по типам</h3>{tokens}"
        f"<h3>Самые дорогие инструменты</h3>{tools}"
        f"<h3>Рост контекста</h3>{growth}<p>Самый дорогой виток: {html.escape(expensive)}</p>"
        f"<h3>По задачам</h3>{tasks}"
    )


def render_timeline_svg(timeline: RunTimeline, width: int = 900, height: int = 320) -> str:
    """Grouped bars per step (input stack and output) plus a context line."""
    steps = timeline.steps
    if not steps:
        return '<p class="muted">Нет событий для timeline.</p>'
    left, right, top, bottom = 60, 20, 20, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    peak = max(max(s.context_tokens, s.output_tokens) for s in steps) or 1
    slot = plot_w / len(steps)
    bar_w = slot * 0.3

    def y(value: float) -> float:
        return top + plot_h - plot_h * value / peak

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = y(peak * fraction)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width - right}" y2="{gy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left - 6}" y="{gy + 4:.1f}" font-size="11" text-anchor="end">{_fmt_int(peak * fraction)}</text>')
    points = []
    for index, step in enumerate(steps):
        x0 = left + slot * index + slot * 0.15
        stack = (
            (step.input_tokens, _COLOR_INPUT),
            (step.cached_input_tokens, _COLOR_CACHED),
            (step.cache_write_input_tokens, _COLOR_WRITE),
        )
        base = 0
        for value, color in stack:
            if value:
                parts.append(
                    f'<rect x="{x0:.1f}" y="{y(base + value):.1f}" width="{bar_w:.1f}" '
                    f'height="{y(base) - y(base + value):.1f}" fill="{color}"><title>{value} tok</title></rect>'
                )
                base += value
        x1 = x0 + bar_w + slot * 0.1
        parts.append(
            f'<rect x="{x1:.1f}" y="{y(step.output_tokens):.1f}" width="{bar_w:.1f}" '
            f'height="{y(0) - y(step.output_tokens):.1f}" fill="{_COLOR_OUTPUT}"><title>{step.output_tokens} tok</title></rect>'
        )
        cx = left + slot * index + slot / 2
        points.append(f"{cx:.1f},{y(step.context_tokens):.1f}")
        parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 16}" font-size="11" text-anchor="middle">виток {step.step}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height - bottom + 30}" font-size="10" text-anchor="middle" fill="#666">{_fmt_money(step.cost)}</text>')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLOR_CONTEXT}" stroke-width="2"/>')
    for point in points:
        px, py = point.split(",")
        parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{_COLOR_CONTEXT}"/>')
    parts.append("</svg>")
    legend = (
        '<p class="legend">'
        f'<span><i style="background:{_COLOR_INPUT}"></i>input (без кэша)</span>'
        f'<span><i style="background:{_COLOR_CACHED}"></i>cached input</span>'
        f'<span><i style="background:{_COLOR_WRITE}"></i>cache write</span>'
        f'<span><i style="background:{_COLOR_OUTPUT}"></i>output</span>'
        f'<span><i style="background:{_COLOR_CONTEXT}"></i>накопленный контекст (всего на входе)</span></p>'
    )
    return "".join(parts) + legend


def _timeline_table(timeline: RunTimeline) -> str:
    return _table(
        ("виток", "input", "cached", "cache write", "output", "контекст", "повторно", "стоимость", "мс")
        + tuple(KIND_TITLES[k] for k in CONTEXT_KINDS),
        [
            (
                str(s.step),
                _fmt_int(s.input_tokens),
                _fmt_int(s.cached_input_tokens),
                _fmt_int(s.cache_write_input_tokens),
                _fmt_int(s.output_tokens),
                _fmt_int(s.context_tokens),
                _fmt_int(s.repeated_tokens),
                _fmt_money(s.cost),
                str(s.latency_ms),
            )
            + tuple(_fmt_int(s.by_kind.get(k, 0.0)) for k in CONTEXT_KINDS)
            for s in timeline.steps
        ],
    )


def _comparison_section(deltas: tuple[Delta, ...], before: str, after: str) -> str:
    rows = []
    for d in deltas:
        if d.unit == "$":
            b, a = _fmt_money(d.before), _fmt_money(d.after)
        elif d.unit == "tok":
            b, a = _fmt_int(d.before), _fmt_int(d.after)
        else:
            b, a = f"{d.before:.1f}%", f"{d.after:.1f}%"
        change = f"{d.delta_points:+.1f} п.п." if d.delta_pct is None and d.unit == "п.п." else (
            "—" if d.delta_pct is None else f"{d.delta_pct:+.1f}%"
        )
        rows.append((d.metric, b, a, change))
    return f"<h2>Сравнение {html.escape(before)} → {html.escape(after)}</h2>" + _table(
        ("метрика", before, after, "изменение"), rows
    )


def choose_timeline_run(events: list[dict], run_id: str = "") -> RunTimeline:
    """The requested run, otherwise the run with the most steps (the most telling picture)."""
    if run_id:
        return build_timeline(events, run_id)
    run_ids = {str(e.get("run_id")) for e in events if e.get("event") == EVENT_LLM_CALL}
    timelines = [build_timeline(events, r) for r in sorted(run_ids)]
    return max(timelines, key=lambda t: (len(t.steps), t.total_cost), default=RunTimeline("", ()))


def render_dashboard(
    summaries: list[Summary],
    timeline: RunTimeline,
    timeline_label: str,
    deltas: tuple[Delta, ...] = (),
) -> str:
    sections = [f"<style>{_CSS}</style><h1>Аудит токенов агента</h1>"]
    sections.append(
        f"<h2>Timeline одного запуска ({html.escape(timeline_label)}, run {html.escape(timeline.run_id[:8])})</h2>"
        + render_timeline_svg(timeline)
        + _timeline_table(timeline)
    )
    if deltas and len(summaries) >= 2:
        sections.append(_comparison_section(deltas, summaries[0].label, summaries[-1].label))
    sections.extend(_summary_section(summary) for summary in summaries)
    return "\n".join(sections)


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_summary(summary: Summary) -> None:
    print(f"\n== {summary.label}: {summary.model} ({summary.provider}), temperature={summary.temperature}, "
          f"задач {summary.task_count} × {summary.runs}, вызовов модели {summary.llm_calls}")
    print(f"success rate: локальные {_fmt_rate(summary.success_rate_local)}, "
          f"сетевые {_fmt_rate(summary.success_rate_network)}")
    print(f"tokens: input {_fmt_int(summary.tokens['input'])}, cached {_fmt_int(summary.tokens['cached'])}, "
          f"cache write {_fmt_int(summary.tokens['cache_write'])}, output {_fmt_int(summary.tokens['output'])}")
    print(f"стоимость: {_fmt_money(summary.cost['total'])} "
          f"(input {_fmt_money(summary.cost['input'])}, output {_fmt_money(summary.cost['output'])}, "
          f"cached {_fmt_money(summary.cost['cached'])}, cache write {_fmt_money(summary.cost['cache_write'])}); "
          f"на задачу {_fmt_money(summary.avg_cost_per_run)}")
    print(f"cache hit rate {_fmt_rate(summary.cache_hit_rate)}; повторно отправлено "
          f"{_fmt_int(summary.repeated_tokens)} tok ({summary.repeated_share * 100:.0f}%)")
    for tool in summary.tools:
        print(f"инструмент {tool.name}: {tool.calls} вызовов, ≈{_fmt_int(tool.estimated_tokens)} tok в контексте")
    if summary.most_expensive_step:
        run_id, step, cost = summary.most_expensive_step
        print(f"самый дорогой виток: run {run_id[:8]} шаг {step} — {_fmt_money(cost)}")
    for kind, value in sorted(summary.growth_per_step.items(), key=lambda kv: -kv[1]):
        print(f"рост {KIND_TITLES.get(kind, kind)}: ≈{_fmt_int(value)} tok/виток")


def _print_deltas(deltas: tuple[Delta, ...]) -> None:
    print("\n== сравнение")
    for d in deltas:
        change = f"{d.delta_points:+.1f} п.п." if d.unit == "п.п." else (
            "—" if d.delta_pct is None else f"{d.delta_pct:+.1f}%"
        )
        print(f"{d.metric:<32} {d.before:>12.4f} → {d.after:>12.4f}  {change}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Отчёт и дашборд по замерам бенчмарка.")
    parser.add_argument("--before", required=True, help="метка замера «до»")
    parser.add_argument("--after", default="", help="метка замера «после» (необязательно)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="куда писать HTML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    labels = [args.before] + ([args.after] if args.after else [])
    measurements = [load_measurement(args.dir, label) for label in labels]
    summaries = [m.summary for m in measurements]
    deltas = compare(summaries[0], summaries[1]) if len(summaries) == 2 else ()
    for summary in summaries:
        _print_summary(summary)
    if deltas:
        _print_deltas(deltas)
    out = args.out or args.dir / DEFAULT_DASHBOARD_NAME
    page = render_dashboard(summaries, measurements[0].timeline, args.before, deltas)
    out.write_text(page, encoding="utf-8")
    print(f"\nДашборд: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
