"""Checks of the agent loop: step limit, tool results, skills, absence of memory."""

import copy

import pytest
from conftest import FakeMessage, make_config

import bot
import harness
import tools
from config import ConfigError
from llms import LLMResult, ToolCall


@pytest.fixture(autouse=True)
def fresh_skills_cache():
    harness.load_skills.cache_clear()
    yield
    harness.load_skills.cache_clear()


@pytest.fixture
def skills_dir(tmp_path):
    directory = tmp_path / "skills"
    directory.mkdir()
    return directory


class ModelDouble:
    """Returns prepared results and records what was sent on every step."""

    def __init__(self, results):
        self.results = list(results)
        self.requests: list[list[dict]] = []
        self.configs: list = []

    async def __call__(self, messages, specs, config):
        self.requests.append(copy.deepcopy(messages))
        self.configs.append(config)
        index = min(len(self.requests) - 1, len(self.results) - 1)
        return self.results[index]


def tool_call(command):
    return ToolCall(id="call_1", name="exec", arguments={"command": command})


def install_tool_double(monkeypatch, output="результат инструмента"):
    calls: list[tuple[str, dict]] = []

    async def fake_run_tool(name, arguments, config):
        calls.append((name, arguments))
        return output

    monkeypatch.setattr(tools, "run_tool", fake_run_tool)
    return calls


async def test_loop_stops_on_the_final_text(monkeypatch):
    model = ModelDouble([LLMResult(text="Готовый ответ")])
    monkeypatch.setattr(harness, "call_llm_step", model)

    result = await harness.run_agent("вопрос", make_config())

    assert result.text == "Готовый ответ"
    assert result.steps == 1
    assert result.limit_reached is False
    assert len(model.requests) == 1


async def test_loop_stops_at_max_steps_and_says_so(monkeypatch):
    model = ModelDouble([LLMResult(text="думаю", tool_calls=(tool_call(["date"]),))])
    monkeypatch.setattr(harness, "call_llm_step", model)
    install_tool_double(monkeypatch)
    config = make_config(agent_max_steps=5)

    result = await harness.run_agent("вопрос", config)

    assert result.limit_reached is True
    assert result.steps == 5
    assert len(model.requests) == 5
    assert "Лимит шагов агента исчерпан (5)" in result.text
    assert "думаю" in result.text


async def test_step_limit_message_reaches_the_user(monkeypatch):
    config = make_config(agent_max_steps=5)
    monkeypatch.setattr(bot, "get_config", lambda: config)
    model = ModelDouble([LLMResult(text="", tool_calls=(tool_call(["date"]),))])
    monkeypatch.setattr(harness, "call_llm_step", model)
    install_tool_double(monkeypatch)
    message = FakeMessage(text="вопрос")

    await bot.handle_text(message)

    assert len(message.answers) == 1
    assert "Лимит шагов агента исчерпан" in message.answers[0]


async def test_tool_result_goes_into_the_next_request(monkeypatch):
    model = ModelDouble(
        [
            LLMResult(tool_calls=(tool_call(["date", "-u"]),)),
            LLMResult(text="Сегодня вторник"),
        ]
    )
    monkeypatch.setattr(harness, "call_llm_step", model)
    calls = install_tool_double(monkeypatch, output="Tue Aug 26 10:00:00 UTC 2026")

    result = await harness.run_agent("какое сегодня число?", make_config())

    assert calls == [("exec", {"command": ["date", "-u"]})]
    second_request = model.requests[1]
    assert second_request[0]["content"] == "какое сегодня число?"
    assert second_request[1]["role"] == "assistant"
    assert second_request[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "exec",
        "content": "Tue Aug 26 10:00:00 UTC 2026",
    }
    assert result.text == "Сегодня вторник"


async def test_skills_reach_the_system_prompt(monkeypatch, skills_dir):
    (skills_dir / "weather.md").write_text(
        "# Погода\nЗапросить wttr.in через exec.", encoding="utf-8"
    )
    model = ModelDouble([LLMResult(text="ок")])
    monkeypatch.setattr(harness, "call_llm_step", model)

    await harness.run_agent("вопрос", make_config(skills_dir=skills_dir))

    system_prompt = model.configs[0].system_prompt
    assert "Запросить wttr.in через exec." in system_prompt
    assert "weather.md" in system_prompt
    assert "exec" in system_prompt


def test_skills_over_the_limit_fail_at_startup(skills_dir):
    (skills_dir / "big.md").write_text("я" * 300, encoding="utf-8")
    (skills_dir / "small.md").write_text("я" * 50, encoding="utf-8")
    config = make_config(skills_dir=skills_dir, skills_max_chars=200)

    with pytest.raises(ConfigError) as error:
        harness.load_skills(config)

    text = str(error.value)
    assert "SKILLS_MAX_CHARS=200" in text
    assert "big.md: 300 симв." in text
    assert "small.md: 50 симв." in text


def test_startup_stops_with_code_one_on_skills_error(monkeypatch):
    monkeypatch.setattr(bot, "get_config", lambda: make_config())

    def fail(config):
        raise ConfigError("слишком много скиллов")

    monkeypatch.setattr(bot, "load_skills", fail)

    assert bot.main() == 1


async def test_no_memory_between_user_messages(monkeypatch):
    model = ModelDouble([LLMResult(text="ответ")])
    monkeypatch.setattr(harness, "call_llm_step", model)
    config = make_config()

    await harness.run_agent("первое сообщение", config)
    await harness.run_agent("второе сообщение", config)

    assert model.requests[0] == [{"role": "user", "content": "первое сообщение"}]
    assert model.requests[1] == [{"role": "user", "content": "второе сообщение"}]
