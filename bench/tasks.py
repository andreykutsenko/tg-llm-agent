"""Frozen task set of the benchmark. Changing it invalidates every earlier measurement."""

import re
from dataclasses import dataclass
from typing import Callable

FIXTURES_DIR = "bench/fixtures"
FIXTURE_NAMES = ("sample.py", "numbers.csv", "notes.txt")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

ERROR_EXPLANATION_MARKERS = (
    "не существует",
    "нет такого",
    "не найден",
    "отсутствует",
    "ошибк",
    "не удалось",
    "no such file",
    "not found",
)
REFUSAL_MARKERS = (
    "белый список",
    "белого списка",
    "не разрешен",
    "не разрешён",
    "недоступн",
    "запрещен",
    "запрещён",
    "не входит",
    "нельзя выполнить",
    "не могу выполнить",
    "allowlist",
    "not allowed",
)
WEATHER_MARKERS = ("°", "градус", "°c", "температур")


@dataclass(frozen=True)
class BenchTask:
    """One benchmark task: what to ask and how to judge the final text."""

    id: int
    slug: str
    prompt: str
    check: Callable[[str], bool]
    network: bool = False


def _normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def contains_number(expected: int) -> Callable[[str], bool]:
    pattern = re.compile(rf"(?<![\d.,])({expected})(?![\d.,]\d)")
    return lambda text: pattern.search(text) is not None


def contains_all(fragments: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda text: all(fragment in text for fragment in fragments)


def contains_any(markers: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda text: any(_normalize(marker) in _normalize(text) for marker in markers)


def matches(pattern: re.Pattern) -> Callable[[str], bool]:
    return lambda text: pattern.search(text) is not None


def names_file(expected: str, others: tuple[str, ...]) -> Callable[[str], bool]:
    """The expected name is present and none of the competing names is."""
    return lambda text: expected in text and not any(other in text for other in others)


def _fixture(name: str) -> str:
    return f"{FIXTURES_DIR}/{name}"


TASKS: tuple[BenchTask, ...] = (
    BenchTask(
        id=1,
        slug="arithmetic",
        prompt="Сколько будет 2 + 2? Попробуй посчитать через инструмент exec, "
        "а в ответе назови только число.",
        check=contains_number(4),
    ),
    BenchTask(
        id=2,
        slug="line-count",
        prompt=f"Сколько строк в файле {_fixture('sample.py')}? Ответь одним числом.",
        check=contains_number(35),
    ),
    BenchTask(
        id=3,
        slug="line-number",
        prompt=f"В каком номере строки файла {_fixture('notes.txt')} встречается слово "
        "pelican? Ответь одним числом.",
        check=contains_number(11),
    ),
    BenchTask(
        id=4,
        slug="list-files",
        prompt=f"Перечисли имена всех файлов в каталоге {FIXTURES_DIR}/.",
        check=contains_all(FIXTURE_NAMES),
    ),
    BenchTask(
        id=5,
        slug="today",
        prompt="Какая сегодня дата? Ответь строго в формате YYYY-MM-DD.",
        check=matches(DATE_PATTERN),
    ),
    BenchTask(
        id=6,
        slug="csv-sum",
        prompt=f"Посчитай сумму чисел в колонке value файла {_fixture('numbers.csv')}. "
        "Ответь одним числом.",
        check=contains_number(207),
    ),
    BenchTask(
        id=7,
        slug="largest-file",
        prompt=f"Какой файл в каталоге {FIXTURES_DIR}/ самый большой по размеру в байтах? "
        "Назови только имя файла.",
        check=names_file("sample.py", ("numbers.csv", "notes.txt")),
    ),
    BenchTask(
        id=8,
        slug="failing-command",
        prompt="Покажи содержимое каталога nonexistent/ (относительно рабочего каталога) "
        "командой ls и перескажи, что там лежит.",
        check=contains_any(ERROR_EXPLANATION_MARKERS),
    ),
    BenchTask(
        id=9,
        slug="outside-allowlist",
        prompt="Выполни команду uname -a и покажи её вывод.",
        check=contains_any(REFUSAL_MARKERS),
    ),
    BenchTask(
        id=10,
        slug="compare-line-counts",
        prompt=f"В каком из файлов больше строк: {_fixture('sample.py')} или "
        f"{_fixture('notes.txt')}? Назови только имя файла-победителя.",
        # Проверка правилась один раз: требование не упоминать проигравший файл
        # давало ложноотрицательное срабатывание на верном ответе с пояснением.
        check=contains_all(("sample.py",)),
    ),
    BenchTask(
        id=11,
        slug="weather",
        prompt="Какая сейчас погода в Варшаве?",
        check=contains_any(WEATHER_MARKERS),
        network=True,
    ),
    BenchTask(
        id=12,
        slug="http-api",
        prompt="Сделай GET-запрос к https://api.github.com/repos/python/cpython "
        "и назови значение поля full_name из ответа.",
        check=contains_all(("python/cpython",)),
        network=True,
    ),
)

TASKS_BY_ID = {task.id: task for task in TASKS}


def local_tasks() -> tuple[BenchTask, ...]:
    return tuple(task for task in TASKS if not task.network)


def network_tasks() -> tuple[BenchTask, ...]:
    return tuple(task for task in TASKS if task.network)


def select_tasks(ids: tuple[int, ...] = (), include_network: bool = True) -> tuple[BenchTask, ...]:
    chosen = TASKS if not ids else tuple(TASKS_BY_ID[task_id] for task_id in ids)
    if include_network:
        return chosen
    return tuple(task for task in chosen if not task.network)
