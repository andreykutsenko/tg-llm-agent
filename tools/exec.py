"""Console command execution with the checks that keep the loop bounded."""

import logging
import os
import subprocess
from pathlib import Path

from config import Config
from llms.protocol import INVALID_ARGUMENTS_KEY, ToolSpec

TOOL_NAME = "exec"
TOOL_DESCRIPTION = (
    "Выполняет консольную команду и возвращает stdout, stderr и код возврата. "
    "Команда передаётся списком аргументов и запускается без шелла: "
    "подстановки через ;, &&, |, кавычки и обратные апострофы не работают. "
    "Разрешены только команды из белого списка, рабочий каталог ограничен."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Команда списком: первый элемент — имя программы, дальше аргументы. "
                'Например ["curl", "-s", "https://wttr.in/Minsk?0"].'
            ),
        }
    },
    "required": ["command"],
}

MINIMAL_ENV_VARS = ("PATH", "LANG")
FALLBACK_PATH = "/usr/local/bin:/usr/bin:/bin"
FALLBACK_LANG = "C.UTF-8"
SECRET_FILE_NAME = ".env"
SECRET_FILE_EXCEPTIONS = (".env.example",)
# Голова и хвост вместо первых N символов: начало показывает структуру,
# конец — итог; пометка говорит модели, сколько именно пропущено.
HEAD_SHARE = 2 / 3
LINE_TRUNCATION_NOTICE = (
    "[... пропущено {skipped} строк из {total} ({chars} символов); "
    "показаны первые {head} и последние {tail} строк ...]"
)
CHAR_TRUNCATION_NOTICE = (
    "[... пропущено {skipped} символов из {total}; показаны начало и конец, "
    "всего {limit} символов ...]"
)

logger = logging.getLogger(__name__)


class CommandRejected(Exception):
    """Command violates a check; the text is meant for the model, not the user."""


def tool_spec() -> ToolSpec:
    return ToolSpec(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, parameters=TOOL_PARAMETERS
    )


def _read_command(arguments: dict) -> list[str]:
    if INVALID_ARGUMENTS_KEY in arguments:
        raise CommandRejected(
            "Отказано: аргументы вызова не разобрались как JSON-объект. "
            'Передайте ровно {"command": ["ls", "-la"]}.'
        )
    command = arguments.get("command")
    if isinstance(command, str):
        raise CommandRejected(
            "Отказано: аргумент command должен быть списком строк, а не строкой. "
            "Команда запускается без шелла, поэтому строку разобрать некому: "
            'передайте ["ls", "-la"] вместо "ls -la".'
        )
    if not isinstance(command, list) or not command:
        raise CommandRejected(
            "Отказано: не передан аргумент command. Ожидается непустой список строк, "
            'например ["date", "-u"].'
        )
    if not all(isinstance(item, str) for item in command):
        raise CommandRejected(
            "Отказано: все элементы command должны быть строками."
        )
    return command


def _check_allowlist(program: str, config: Config) -> None:
    allowed = ", ".join(config.exec_allowlist)
    if os.sep in program or (os.altsep and os.altsep in program):
        raise CommandRejected(
            f"Отказано: команда {program!r} задана путём. Указывайте только имя "
            f"программы из белого списка: {allowed}."
        )
    if program not in config.exec_allowlist:
        raise CommandRejected(
            f"Отказано: команда {program!r} не входит в белый список. "
            f"Разрешены только: {allowed}. Решите задачу этими командами "
            "или объясните пользователю, что она недоступна."
        )


def _is_secret_file(path: Path) -> bool:
    name = path.name
    if name in SECRET_FILE_EXCEPTIONS:
        return False
    return name == SECRET_FILE_NAME or name.startswith(f"{SECRET_FILE_NAME}.")


def _check_argument_paths(arguments: list[str], config: Config) -> None:
    for argument in arguments:
        if argument.startswith("-") or "://" in argument:
            continue
        candidate = Path(argument).expanduser()
        if not candidate.is_absolute():
            candidate = config.exec_workdir / candidate
        resolved = Path(os.path.normpath(str(candidate)))
        if _is_secret_file(resolved):
            raise CommandRejected(
                f"Отказано: файл {argument!r} содержит секреты (токен Telegram, "
                "ключ API) и недоступен. Не пытайтесь читать его другим способом."
            )
        if not resolved.is_relative_to(config.exec_workdir):
            raise CommandRejected(
                f"Отказано: путь {argument!r} ведёт за пределы рабочего каталога "
                f"{config.exec_workdir}. Работайте только внутри него."
            )


def _minimal_env() -> dict[str, str]:
    """Explicit minimum: the bot's own environment holds the bot token and API key."""
    defaults = {"PATH": FALLBACK_PATH, "LANG": FALLBACK_LANG}
    return {name: os.environ.get(name) or defaults[name] for name in MINIMAL_ENV_VARS}


def _split_head_tail(limit: int) -> tuple[int, int]:
    head = max(1, round(limit * HEAD_SHARE))
    return (head, max(limit - head, 0))


def _clip_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head, tail = _split_head_tail(max_lines)
    skipped = lines[head : len(lines) - tail] if tail else lines[head:]
    notice = LINE_TRUNCATION_NOTICE.format(
        skipped=len(skipped),
        total=len(lines),
        chars=sum(len(line) + 1 for line in skipped),
        head=head,
        tail=tail,
    )
    kept_tail = lines[len(lines) - tail :] if tail else []
    return "\n".join(lines[:head] + [notice] + kept_tail)


def _clip_chars(text: str, budget: int) -> tuple[str, int]:
    if len(text) <= budget:
        return (text, len(text))
    head, tail = _split_head_tail(budget)
    notice = CHAR_TRUNCATION_NOTICE.format(
        skipped=len(text) - head - tail, total=len(text), limit=budget
    )
    kept_tail = text[len(text) - tail :] if tail else ""
    return (f"{text[:head]}\n{notice}\n{kept_tail}", budget)


def _clip(text: str, budget: int, max_lines: int) -> tuple[str, int]:
    """Lines first, then characters; the caller gets how much of the budget was used."""
    if budget <= 0:
        return ("", 0)
    return _clip_chars(_clip_lines(text, max_lines), budget)


def _render(completed: subprocess.CompletedProcess, config: Config) -> str:
    stdout, used = _clip(completed.stdout or "", config.exec_max_output, config.exec_max_lines)
    stderr, _ = _clip(
        completed.stderr or "", config.exec_max_output - used, config.exec_max_lines
    )
    return (
        f"exit_code: {completed.returncode}\n"
        f"stdout:\n{stdout or '(пусто)'}\n"
        f"stderr:\n{stderr or '(пусто)'}"
    )


def run_exec(arguments: dict, config: Config) -> str:
    """Run a console command; every refusal is a normal text answer to the model."""
    try:
        command = _read_command(arguments)
        _check_allowlist(command[0], config)
        _check_argument_paths(command[1:], config)
    except CommandRejected as rejection:
        logger.info("exec отклонён: %s | аргументы: %s", rejection, arguments)
        return str(rejection)

    logger.info(
        "exec запускает %s в %s (таймаут %.0f с)",
        command,
        config.exec_workdir,
        config.exec_timeout_seconds,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(config.exec_workdir),
            env=_minimal_env(),
            capture_output=True,
            text=True,
            timeout=config.exec_timeout_seconds,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("exec снял команду %s по таймауту", command)
        return (
            f"Команда не уложилась в {config.exec_timeout_seconds:.0f} секунд "
            "и была снята. Попробуйте более быструю команду или уточните запрос."
        )
    except FileNotFoundError:
        return (
            f"Команда {command[0]!r} разрешена, но не установлена в системе. "
            "Выберите другую команду."
        )
    except OSError as error:
        return f"Команду не удалось запустить: {error}."

    return _render(completed, config)
