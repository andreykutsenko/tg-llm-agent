"""Token audit: usage extraction, recorder, benchmark verdicts, aggregation, comparison."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
from conftest import FakeClient, anthropic_response, make_config, openai_response

import harness
import llms
import tools
from bench import report
from bench.pricing import cost_breakdown
from bench.run import (
    BenchMeta,
    TaskOutcome,
    bench_config,
    build_measurement,
    judge,
    run_benchmark,
    write_measurement,
)
from bench.tasks import TASKS_BY_ID, local_tasks, network_tasks
from config import PROJECT_ROOT
from harness import AgentResult
from llms import anthropic as anthropic_provider
from llms import ollama as ollama_provider
from llms.protocol import LLMResult, ToolCall, Usage
from observability import recorder
from observability.recorder import EVENT_LLM_CALL, EVENT_TOOL_CALL


@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder.configure(path)
    harness.load_skills.cache_clear()
    yield path
    recorder.configure(None)
    harness.load_skills.cache_clear()


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# 1. Usage корректно извлекается из ответа Anthropic (в т.ч. cache_read)
def test_anthropic_usage_is_extracted_with_cache_read():
    response = anthropic_response("ok")
    response.usage = SimpleNamespace(
        input_tokens=1200, output_tokens=80, cache_read_input_tokens=900, cache_creation_input_tokens=50
    )

    usage = anthropic_provider.parse_response(response).usage

    assert usage == Usage(
        input_tokens=1200, output_tokens=80, cached_input_tokens=900, cache_write_input_tokens=50
    )


# 2. Usage у Ollama маппится из prompt_eval_count / eval_count
@pytest.mark.parametrize(
    "usage_block",
    [
        SimpleNamespace(prompt_eval_count=310, eval_count=42),
        SimpleNamespace(prompt_tokens=310, completion_tokens=42),
    ],
)
def test_ollama_usage_is_mapped_from_native_and_openai_fields(usage_block):
    response = openai_response("ответ")
    response.usage = usage_block

    usage = ollama_provider.parse_response(response).usage

    assert usage.input_tokens == 310
    assert usage.output_tokens == 42
    assert usage.cached_input_tokens == 0
    assert usage.reasoning_tokens == 0


def test_reasoning_tokens_are_read_when_the_provider_reports_them():
    response = openai_response("ответ")
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=50, completion_tokens_details=SimpleNamespace(reasoning_tokens=30)
    )

    assert ollama_provider.parse_response(response).usage.reasoning_tokens == 30


def test_anthropic_thinking_tokens_go_to_reasoning_tokens():
    response = anthropic_response("ok")
    response.usage = SimpleNamespace(
        input_tokens=5, output_tokens=500, output_tokens_details=SimpleNamespace(thinking_tokens=420)
    )
    without_details = anthropic_response("ok")
    without_details.usage = SimpleNamespace(input_tokens=5, output_tokens=500)

    usage = anthropic_provider.parse_response(response).usage

    assert (usage.output_tokens, usage.reasoning_tokens) == (500, 420)
    assert anthropic_provider.parse_response(without_details).usage.reasoning_tokens == 0


async def test_anthropic_gets_effort_but_never_temperature(monkeypatch):
    client = FakeClient(response=anthropic_response("ok"))
    monkeypatch.setattr(anthropic_provider, "AsyncAnthropic", lambda **kwargs: client)
    base = make_config(
        llm_provider="anthropic", llm_api_key="sk-ant-test", llm_model="claude-opus-5", llm_temperature=0.0
    )

    await llms.call_llm("вопрос", config=base)
    await llms.call_llm("вопрос", config=bench_config(base, "claude-haiku-4-5", "low"))

    assert "temperature" not in client.calls[0] and "output_config" not in client.calls[0]
    assert "temperature" not in client.calls[1]
    assert client.calls[1]["output_config"] == {"effort": "low"}
    assert client.calls[1]["model"] == "claude-haiku-4-5"


async def test_ollama_still_gets_temperature_when_configured(monkeypatch):
    client = FakeClient(response=openai_response("ok"))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: client)

    await llms.call_llm("вопрос", config=make_config(llm_temperature=0.0))

    assert client.calls[0]["temperature"] == 0.0


# 3. Провайдер не вернул usage → нули, исключения нет
async def test_missing_usage_gives_zeros_without_raising(monkeypatch):
    client = FakeClient(response=openai_response("ответ без usage"))
    monkeypatch.setattr(ollama_provider, "AsyncOpenAI", lambda **kwargs: client)

    result = await llms.call_llm_step([llms.user_message("вопрос")], (), make_config())

    assert result.text == "ответ без usage"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (0, 0)
    assert result.usage.is_reported is False


# 4. record_llm_call пишет ровно одну валидную JSON-строку на вызов
def test_record_llm_call_writes_exactly_one_json_line(isolated_telemetry):
    usage = Usage(input_tokens=10, output_tokens=5, cached_input_tokens=3, latency_ms=120)

    recorder.record_llm_call("run-1", 2, "claude-opus-5", usage, 0.001, ({"kind": "system", "chars": 40, "hash": "a"},))

    lines = isolated_telemetry.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == EVENT_LLM_CALL
    assert (event["run_id"], event["step"], event["model"]) == ("run-1", 2, "claude-opus-5")
    assert (event["input_tokens"], event["output_tokens"], event["cached_input_tokens"]) == (10, 5, 3)
    assert event["usage_missing"] is False
    assert event["context_blocks"][0]["kind"] == "system"


async def test_agent_loop_records_llm_and_tool_events_with_one_run_id(monkeypatch, isolated_telemetry):
    call = ToolCall(id="c1", name="exec", arguments={"command": ["date"]})
    results = [
        LLMResult(tool_calls=(call,), usage=Usage(input_tokens=100, output_tokens=20)),
        LLMResult(text="готово", usage=Usage(input_tokens=150, output_tokens=10)),
    ]

    requests: list = []

    async def fake_generate(messages, specs, config):
        requests.append(messages)
        return results[min(len(requests) - 1, len(results) - 1)]

    monkeypatch.setattr(llms, "_STEP_GENERATORS", {"ollama": fake_generate})
    monkeypatch.setattr(tools, "REGISTRY", (tools.Tool(spec=tools.exec_tool.tool_spec(), run=lambda a, c: "Tue"),))
    monkeypatch.setattr(tools, "_BY_NAME", {t.spec.name: t for t in tools.REGISTRY})

    result = await harness.run_agent("который час?", make_config())

    events = read_events(isolated_telemetry)
    assert result.text == "готово"
    assert [e["event"] for e in events] == [EVENT_LLM_CALL, EVENT_TOOL_CALL, EVENT_LLM_CALL]
    assert len({e["run_id"] for e in events}) == 1
    assert [e["step"] for e in events] == [1, 1, 2]
    assert events[1]["tool_name"] == "exec"
    assert events[1]["output_size"] == 3
    assert events[2]["input_tokens"] == 150
    kinds = [b["kind"] for b in events[2]["context_blocks"]]
    assert kinds == ["system", "tools", "user", "assistant", "tool"]
    assert events[0]["increment"] is None
    assert events[2]["increment"]["method"] == "chars" and events[2]["increment"]["tool_chars"] == 3


# 5. Падение записи телеметрии не роняет run_agent
async def test_telemetry_failure_does_not_break_the_agent(monkeypatch, tmp_path, caplog):
    recorder.configure(tmp_path / "not-a-dir.jsonl" / "events.jsonl")
    (tmp_path / "not-a-dir.jsonl").write_text("файл, а не каталог", encoding="utf-8")

    async def fake_generate(messages, specs, config):
        return LLMResult(text="ответ", usage=Usage(input_tokens=1, output_tokens=1))

    monkeypatch.setattr(llms, "_STEP_GENERATORS", {"ollama": fake_generate})

    result = await harness.run_agent("вопрос", make_config())

    assert result.text == "ответ"
    assert "Телеметрия не записана" in caplog.text


# 6. Счётчик повторных токенов на синтетическом прогоне из трёх витков
def synthetic_run(run_id="run-x", model="claude-opus-5"):
    system = {"kind": "system", "chars": 800, "hash": "sys"}
    user = {"kind": "user", "chars": 100, "hash": "usr"}
    assistant_1 = {"kind": "assistant", "chars": 50, "hash": "a1"}
    tool_1 = {"kind": "tool", "chars": 50, "hash": "t1"}
    assistant_2 = {"kind": "assistant", "chars": 100, "hash": "a2"}
    tool_2 = {"kind": "tool", "chars": 100, "hash": "t2"}

    def llm(step, blocks, input_tokens, output_tokens, cached=0, increment=None):
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached)
        return {
            "event": EVENT_LLM_CALL, "run_id": run_id, "step": step, "model": model,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_input_tokens": cached,
            "cache_write_input_tokens": 0, "latency_ms": 100 * step,
            "cost": cost_breakdown(model, usage).total, "usage_missing": False, "context_blocks": blocks,
            "increment": increment,
        }

    def counted(assistant_tokens, tool_tokens, tool_chars):
        return {
            "method": "count_tokens", "assistant_chars": assistant_tokens, "tool_chars": tool_chars,
            "assistant_tokens": assistant_tokens, "tool_tokens": tool_tokens,
            "tool_messages": [{"name": "exec", "chars": tool_chars}],
        }

    def tool(step, size):
        return {"event": EVENT_TOOL_CALL, "run_id": run_id, "step": step, "tool_name": "exec",
                "input_size": 20, "output_size": size, "duration_ms": 5}

    return [
        llm(1, [system, user], 900, 50),
        tool(1, 50),
        llm(2, [system, user, assistant_1, tool_1], 1000, 100, increment=counted(30, 70, 50)),
        tool(2, 100),
        llm(3, [system, user, assistant_1, tool_1, assistant_2, tool_2], 1200, 30, increment=counted(50, 150, 100)),
    ]


def test_repeated_tokens_on_a_synthetic_three_step_run():
    events = synthetic_run()

    repeated, total = report.count_repeated_tokens(report.llm_events_of_run(events, "run-x"))

    # виток 1: ничего; виток 2: весь payload витка 1 = 900; виток 3: весь payload витка 2 = 1000
    assert total == 3100
    assert repeated == 1900 and isinstance(repeated, int)


def test_growth_is_the_exact_difference_and_split_by_counted_ratio():
    timeline = report.build_timeline(synthetic_run(), "run-x")

    assert [s.growth_tokens for s in timeline.steps] == [0, 100, 200]
    # виток 2: 100 tok в пропорции 30:70; виток 3: 200 tok в пропорции 50:150
    assert (timeline.steps[1].assistant_growth, timeline.steps[1].tool_growth) == (30, 70)
    assert (timeline.steps[2].assistant_growth, timeline.steps[2].tool_growth) == (50, 150)
    assert timeline.steps[2].increment_method == "count_tokens"


def test_growth_falls_back_to_chars_inside_the_increment_only():
    by_chars = {"method": "chars", "assistant_chars": 25, "tool_chars": 75, "assistant_tokens": None, "tool_tokens": None}

    assert report.split_growth(200, by_chars) == (50, 150, "chars")
    assert report.split_growth(200, None) == (200, 0, "none")
    assert report.split_growth(0, by_chars) == (0, 0, "chars")


async def test_increment_is_described_from_the_messages_appended_since_the_previous_step():
    from observability import remember_payload, set_current_step

    set_current_step("run-inc", 2)
    call = ToolCall(id="c1", name="exec", arguments={"command": ["ls"]})
    messages = [
        llms.user_message("вопрос"),
        llms.assistant_message(LLMResult(text="смотрю", tool_calls=(call,))),
        llms.tool_message(call, "a.txt\nb.txt"),
    ]
    remember_payload("run-inc", 1)

    increment = await llms.describe_increment("run-inc", messages, make_config())

    assert increment["method"] == "chars"
    assert increment["tool_chars"] == len("a.txt\nb.txt")
    assert increment["assistant_chars"] > len("смотрю")
    assert increment["tool_messages"] == [{"name": "exec", "chars": len("a.txt\nb.txt")}]
    assert await llms.describe_increment("run-other", messages, make_config()) is None


# 7. Проверка задачи бенчмарка: правильный ответ → success, неправильный → fail
def test_task_check_accepts_right_answer_and_rejects_wrong():
    task = TASKS_BY_ID[2]

    right = judge(task, AgentResult(text="В файле 35 строк.", steps=2, limit_reached=False), "")
    wrong = judge(task, AgentResult(text="В файле 36 строк.", steps=2, limit_reached=False), "")
    crashed = judge(task, None, "LLMError: сбой")

    assert right == (True, True)
    assert wrong == (False, False)
    assert crashed == (False, False)


# 8. limit_reached == True делает задачу failed, даже если текст похож на верный
def test_step_limit_makes_task_failed_despite_plausible_text():
    task = TASKS_BY_ID[2]

    check_passed, success = judge(task, AgentResult(text="35", steps=8, limit_reached=True), "")

    assert check_passed is True
    assert success is False


def test_task_set_is_frozen_and_fixture_numbers_match_files():
    fixtures = list((PROJECT_ROOT / "bench" / "fixtures").iterdir())
    names = sorted(p.name for p in fixtures)
    sample = next(p for p in fixtures if p.name == "sample.py")
    notes = next(p for p in fixtures if p.name == "notes.txt")
    numbers = next(p for p in fixtures if p.name == "numbers.csv")
    csv_sum = sum(int(line) for line in numbers.read_text().splitlines()[1:] if line.strip())
    pelican_line = next(i for i, line in enumerate(notes.read_text().splitlines(), 1) if "pelican" in line)

    assert len(local_tasks()) == 10 and len(network_tasks()) == 2
    assert names == ["notes.txt", "numbers.csv", "sample.py"]
    assert TASKS_BY_ID[2].check(f"{len(sample.read_text().splitlines())}")
    assert TASKS_BY_ID[3].check(f"{pelican_line}")
    assert TASKS_BY_ID[6].check(f"{csv_sum}")
    assert max(fixtures, key=lambda p: p.stat().st_size).name == "sample.py"
    assert TASKS_BY_ID[8].check("Каталог nonexistent не существует, ls вернул ошибку.")
    assert not TASKS_BY_ID[8].check("В каталоге лежат файлы a.txt и b.txt.")
    assert TASKS_BY_ID[9].check("Команда uname не входит в белый список, выполнить её нельзя.")
    assert TASKS_BY_ID[5].check("Сегодня 2026-09-03.")
    assert TASKS_BY_ID[10].check("sample.py (35 строк против 13 у notes.txt).")
    assert not TASKS_BY_ID[10].check("notes.txt")
    assert not TASKS_BY_ID[5].check("Сегодня третье сентября.")


async def test_failed_task_is_reported_without_stopping_the_run(monkeypatch):
    calls = []

    async def flaky_agent(prompt, config):
        calls.append(prompt)
        if len(calls) == 1:
            raise llms.LLMError("API недоступен")
        return AgentResult(text="35", steps=1, limit_reached=False)

    monkeypatch.setattr("bench.run.run_agent", flaky_agent)
    tasks = (TASKS_BY_ID[2], TASKS_BY_ID[2])

    outcomes = await run_benchmark(tasks, runs=1, config=make_config())

    assert [o.success for o in outcomes] == [False, True]
    assert outcomes[0].error == "LLMError: API недоступен"
    assert len(calls) == 2


# 9. Агрегация стоимости раздельно по input / output / cached
def test_cost_is_aggregated_separately_by_token_type(tmp_path):
    events = synthetic_run()
    outcome = TaskOutcome(2, "line-count", False, 1, "run-x", True, True, False, 3, "", "35", 900)
    outcomes = [dataclasses.asdict(outcome)]

    summary = report.aggregate({"label": "before", "model": "claude-opus-5", "runs": 1}, outcomes, events)

    assert summary.tokens == {"input": 3100, "output": 180, "cached": 0, "cache_write": 0, "reasoning": 0}
    assert summary.cost["input"] == pytest.approx(3100 * 5.0 / 1_000_000)
    assert summary.cost["output"] == pytest.approx(180 * 25.0 / 1_000_000)
    assert summary.cost["cached"] == 0.0
    assert summary.cost["total"] == pytest.approx(summary.cost["input"] + summary.cost["output"])
    assert summary.success_rate_local == 1.0
    assert summary.tools[0].name == "exec" and summary.tools[0].calls == 2
    assert summary.tools[0].estimated_tokens == pytest.approx(70.0 + 150.0)
    # виток 2: 1000×$5 + 100×$25 = $0.0075 дороже витка 3: 1200×$5 + 30×$25 = $0.00675
    assert summary.most_expensive_step[1] == 2
    assert summary.growth_per_step["system"] == 0.0 and summary.growth_per_step["user"] == 0.0
    assert summary.growth_per_step["tool"] == pytest.approx((70 + 150) / 2)
    assert summary.growth_per_step["assistant"] == pytest.approx((30 + 50) / 2)
    assert summary.increment_methods == {"count_tokens": 2}


def test_context_constants_come_from_meta_and_first_step():
    meta = {"label": "x", "model": "claude-opus-5", "runs": 1, "system_prompt_tokens": 700, "tool_descriptions_tokens": 100}

    summary = report.aggregate(meta, [], synthetic_run())

    # виток 1 читает 900 tok: 700 system + 100 tools + 100 постановка задачи
    assert summary.context_constants == {"system": 700.0, "tools": 100.0, "user": 100.0}


def test_measurement_json_keeps_model_effort_and_round_trips(tmp_path):
    meta = BenchMeta("before", "anthropic", "claude-haiku-4-5", "low", 2, (2,), 8, 4000, 0.0)
    outcomes = [
        TaskOutcome(2, "line-count", False, 1, "run-x", True, True, False, 3, "", "35", 900),
        TaskOutcome(2, "line-count", False, 2, "", False, False, False, 0, "LLMError: сбой", "", 10),
    ]
    path = tmp_path / "results" / "before.json"

    write_measurement(path, build_measurement(meta, outcomes, synthetic_run()))
    loaded = report.load_measurement(tmp_path / "results", "before")

    assert loaded.meta["model"] == "claude-haiku-4-5" and loaded.meta["effort"] == "low"
    assert loaded.summary.model == "claude-haiku-4-5" and loaded.summary.effort == "low"
    assert loaded.summary.success_rate_local == 0.5
    assert loaded.summary.success_rate_by_run == (1.0, 0.0)
    assert loaded.summary.cost["total"] == pytest.approx(3100 * 5e-6 + 180 * 25e-6)
    assert loaded.summary.tools[0].name == "exec"
    assert loaded.summary.task_runs == {2: ("run-x",)}
    assert loaded.timeline.run_id == "run-x" and len(loaded.timeline.steps) == 3
    assert "answer_excerpt" not in path.read_text(encoding="utf-8")


def test_cached_tokens_are_priced_at_the_cache_rate():
    breakdown = cost_breakdown(
        "claude-opus-5", Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=1000)
    )

    assert breakdown.input == pytest.approx(0.005)
    assert breakdown.cached_input == pytest.approx(0.0005)


# 10. report сравнивает два замера и считает дельту в процентах
def test_report_compares_two_measurements_in_percent():
    before_events = synthetic_run("run-b")
    after_events = synthetic_run("run-a")
    for event in after_events:
        if event["event"] == EVENT_LLM_CALL:
            event["input_tokens"] = event["input_tokens"] // 2
            event["cost"] = event["cost"] / 2
    outcomes = [{"kind": "outcome", "task_id": 2, "slug": "x", "network": False, "success": True, "run_id": "run-b", "steps": 3}]

    before = report.aggregate({"label": "before", "model": "claude-opus-5", "runs": 1}, outcomes, before_events)
    after = report.aggregate({"label": "after", "model": "claude-opus-5", "runs": 1}, outcomes, after_events)
    deltas = {d.metric: d for d in report.compare(before, after)}

    assert deltas["input tokens"].delta_pct == pytest.approx(-50.0)
    assert deltas["стоимость output"].delta_pct == pytest.approx(0.0)
    assert deltas["стоимость, всего"].delta_pct < 0
    assert deltas["success rate (локальные)"].delta_points == pytest.approx(0.0)


def test_dashboard_contains_timeline_of_one_run_and_comparison():
    events = synthetic_run()
    outcomes = [{"kind": "outcome", "task_id": 2, "slug": "x", "network": False, "success": True, "run_id": "run-x", "steps": 3}]
    summary = report.aggregate({"label": "before", "model": "claude-opus-5", "runs": 1}, outcomes, events)
    timeline = report.choose_timeline_run(events)

    page = report.render_dashboard([summary, summary], timeline, "before", report.compare(summary, summary))

    assert timeline.run_id == "run-x" and len(timeline.steps) == 3
    assert "<svg" in page and "виток 3" in page
    assert "Сравнение before → before" in page
    assert "cache hit rate" in page


async def test_anthropic_marks_system_and_last_tool_for_caching(monkeypatch):
    client = FakeClient(response=anthropic_response("ok"))
    monkeypatch.setattr(anthropic_provider, "AsyncAnthropic", lambda **kwargs: client)
    config = make_config(llm_provider="anthropic", llm_api_key="sk-ant-test", system_prompt="Системный")

    await llms.call_llm_step([llms.user_message("вопрос")], tools.tool_specs(), config)

    request = client.calls[0]
    assert request["system"] == [{"type": "text", "text": "Системный", "cache_control": {"type": "ephemeral"}}]
    assert request["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert request["messages"] == [{"role": "user", "content": "вопрос"}]
