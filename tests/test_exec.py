"""Checks of the exec tool: no shell, allowlist, workdir, clean env, limits."""

import subprocess
from types import SimpleNamespace

import pytest
from conftest import make_config

from tools import exec as exec_tool


@pytest.fixture
def recorded_run(monkeypatch):
    """Replace the real subprocess with a double that records how it was called."""
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def forbidden_run(monkeypatch):
    def fake_run(command, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError(f"команда не должна была запуститься: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_command_outside_allowlist_is_rejected(forbidden_run):
    answer = exec_tool.run_exec({"command": ["rm", "-rf", "data"]}, make_config())

    assert "не входит в белый список" in answer
    assert "rm" in answer


def test_semicolon_argument_stays_a_plain_string(recorded_run):
    answer = exec_tool.run_exec(
        {"command": ["ls", "; rm -rf /"]}, make_config()
    )

    call = recorded_run[0]
    assert call["command"] == ["ls", "; rm -rf /"]
    assert call["shell"] is False
    assert "exit_code: 0" in answer


def test_command_as_string_is_rejected(forbidden_run):
    answer = exec_tool.run_exec({"command": "ls -la"}, make_config())

    assert "списком строк" in answer


def test_timeout_kills_the_process_and_explains_it(monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    answer = exec_tool.run_exec({"command": ["curl", "https://example.com"]}, make_config())

    assert "не уложилась" in answer
    assert "снята" in answer


def test_subprocess_does_not_inherit_bot_environment(monkeypatch, recorded_run):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    exec_tool.run_exec({"command": ["date"]}, make_config())

    env = recorded_run[0]["env"]
    assert set(env) == {"PATH", "LANG"}
    assert "123456:secret-token" not in "".join(env.values())
    assert "sk-ant-secret" not in "".join(env.values())


def test_reading_dotenv_is_rejected(forbidden_run):
    answer = exec_tool.run_exec({"command": ["cat", ".env"]}, make_config())

    assert "секреты" in answer


def test_path_outside_workdir_is_rejected(forbidden_run):
    answer = exec_tool.run_exec({"command": ["cat", "/etc/passwd"]}, make_config())

    assert "за пределы рабочего каталога" in answer


def test_long_output_is_truncated_with_a_notice(monkeypatch):
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="я" * 5000, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = make_config(exec_max_output=100)

    answer = exec_tool.run_exec({"command": ["ls"]}, config)

    assert "пропущено 4900 символов из 5000" in answer
    assert answer.count("я") == 100
    assert answer.index("пропущено") > answer.index("я" * 67)


def test_multiline_output_keeps_head_and_tail_with_skipped_count(monkeypatch):
    lines = [f"line {i}" for i in range(1, 101)]

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = make_config(exec_max_lines=30)

    answer = exec_tool.run_exec({"command": ["cat", "big.txt"]}, config)

    assert "line 20\n" in answer and "line 91\n" in answer and "line 100" in answer
    assert "line 21\n" not in answer and "line 90\n" not in answer
    assert "пропущено 70 строк из 100" in answer
    assert "показаны первые 20 и последние 10 строк" in answer


def test_short_output_is_returned_untouched(monkeypatch):
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="a\nb\nc", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    answer = exec_tool.run_exec({"command": ["ls"]}, make_config(exec_max_lines=30))

    assert "stdout:\na\nb\nc\n" in answer
    assert "пропущено" not in answer


def test_command_runs_in_the_configured_workdir(recorded_run):
    config = make_config()

    exec_tool.run_exec({"command": ["ls"]}, config)

    assert recorded_run[0]["cwd"] == str(config.exec_workdir)
    assert recorded_run[0]["timeout"] == config.exec_timeout_seconds


def test_unparsable_arguments_are_rejected_with_an_example(forbidden_run):
    from llms.protocol import INVALID_ARGUMENTS_KEY

    answer = exec_tool.run_exec({INVALID_ARGUMENTS_KEY: "{command: ls"}, make_config())

    assert "не разобрались как JSON" in answer
