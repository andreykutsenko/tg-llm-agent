"""Agent loop: model asks for a tool → code runs it → result goes back to the model."""

import dataclasses
import logging
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache

import tools
from config import Config, ConfigError
from llms import assistant_message, call_llm_step, tool_message, user_message
from observability import set_current_step

SKILLS_SUFFIX = ".md"
# Скилл со строкой «<!-- triggers: погод, weather -->» уходит в system только
# когда запрос содержит один из триггеров; без строки — всегда.
SKILL_TRIGGERS = re.compile(r"<!--\s*triggers:\s*(.*?)\s*-->", re.IGNORECASE | re.DOTALL)
AGENT_INSTRUCTIONS = (
    "Ты — агент с доступом к инструментам. Если для ответа нужны данные, "
    "которых у тебя нет, вызывай инструмент, а не выдумывай результат. "
    "Инструмент может отказать: прочитай причину отказа и попробуй иначе "
    "или объясни пользователю, почему задача невыполнима. "
    "Когда данных достаточно, дай короткий финальный ответ текстом."
)
TOOLS_HEADER = "Доступные инструменты:"
SKILLS_HEADER = (
    "Скиллы — инструкции, как выполнять типовые задачи. Это не инструменты: "
    "действия выполняй через инструменты, скилл лишь объясняет, что именно делать."
)
STEP_LIMIT_NOTICE = (
    "\n\n⚠️ Лимит шагов агента исчерпан ({max_steps}), ответ может быть неполным."
)
NO_PARTIAL_ANSWER_TEXT = "Агент не успел сформулировать ответ."

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    """One skills/*.md: its text for the model and the words that summon it."""

    name: str
    text: str
    triggers: tuple[str, ...]

    def matches(self, user_text: str) -> bool:
        if not self.triggers:
            return True
        lowered = user_text.lower()
        return any(trigger in lowered for trigger in self.triggers)


@dataclass(frozen=True)
class AgentResult:
    """What the loop produced and how it ended."""

    text: str
    steps: int
    limit_reached: bool


def _describe_sizes(files, sizes) -> str:
    return "\n".join(f"  {path.name}: {size} симв." for path, size in zip(files, sizes))


def _parse_skill(path, raw: str) -> Skill:
    match = SKILL_TRIGGERS.search(raw)
    triggers = ()
    if match:
        triggers = tuple(
            item.strip().lower() for item in match.group(1).split(",") if item.strip()
        )
        raw = SKILL_TRIGGERS.sub("", raw, count=1)
    return Skill(name=path.name, text=raw.strip(), triggers=triggers)


@lru_cache(maxsize=1)
def load_skills(config: Config) -> tuple[Skill, ...]:
    """Read every skills/*.md once at start; oversize is a startup error."""
    directory = config.skills_dir
    if not directory.is_dir():
        logger.warning("Каталог скиллов %s не найден, скиллы не загружены", directory)
        return ()
    files = sorted(path for path in directory.glob(f"*{SKILLS_SUFFIX}") if path.is_file())
    if not files:
        logger.warning("В каталоге %s нет файлов %s", directory, SKILLS_SUFFIX)
        return ()
    contents = [path.read_text(encoding="utf-8") for path in files]
    sizes = [len(text) for text in contents]
    total = sum(sizes)
    if total > config.skills_max_chars:
        raise ConfigError(
            f"Суммарный размер скиллов {total} символов превышает SKILLS_MAX_CHARS="
            f"{config.skills_max_chars}. Скиллы уходят в каждый запрос, "
            f"поэтому обрезать их молча нельзя. Файлы:\n"
            f"{_describe_sizes(files, sizes)}"
        )
    logger.info("Загружено скиллов: %d, суммарно %d символов", len(files), total)
    return tuple(_parse_skill(path, text) for path, text in zip(files, contents))


def select_skills(skills: tuple[Skill, ...], user_text: str) -> tuple[Skill, ...]:
    """Only the skills the request calls for; the rest stay out of the system prompt."""
    return tuple(skill for skill in skills if skill.matches(user_text))


def render_skills(skills: tuple[Skill, ...]) -> str:
    return "\n\n".join(f"### {skill.name}\n{skill.text}" for skill in skills)


def build_system_prompt(config: Config, skills: str) -> str:
    """Base prompt plus agent rules, tool listing and all skills."""
    sections = [
        config.system_prompt,
        AGENT_INSTRUCTIONS,
        f"{TOOLS_HEADER}\n{tools.describe_tools()}",
    ]
    if skills:
        sections.append(f"{SKILLS_HEADER}\n\n{skills}")
    return "\n\n".join(sections)


def _agent_config(config: Config, skills: str) -> Config:
    return dataclasses.replace(config, system_prompt=build_system_prompt(config, skills))


def _finish_with_limit(partial_texts: list[str], config: Config) -> AgentResult:
    body = "\n\n".join(text for text in partial_texts if text) or NO_PARTIAL_ANSWER_TEXT
    logger.warning("Лимит шагов %d исчерпан, отдаём частичный ответ", config.agent_max_steps)
    return AgentResult(
        text=body + STEP_LIMIT_NOTICE.format(max_steps=config.agent_max_steps),
        steps=config.agent_max_steps,
        limit_reached=True,
    )


async def run_agent(user_text: str, config: Config) -> AgentResult:
    """Run the loop for exactly one user message; the context dies with the answer."""
    skills = select_skills(load_skills(config), user_text)
    logger.info("Скиллы для запроса: %s", ", ".join(s.name for s in skills) or "нет")
    step_config = _agent_config(config, render_skills(skills))
    specs = tools.tool_specs()
    messages = [user_message(user_text)]
    partial_texts: list[str] = []
    run_id = str(uuid.uuid4())

    for step in range(1, config.agent_max_steps + 1):
        set_current_step(run_id, step)
        result = await call_llm_step(messages, specs, step_config)
        if not result.wants_tools:
            logger.info("Шаг %d/%d: финальный ответ модели", step, config.agent_max_steps)
            return AgentResult(text=result.text, steps=step, limit_reached=False)
        if result.text:
            partial_texts.append(result.text)
        messages.append(assistant_message(result))
        for call in result.tool_calls:
            output = await tools.run_tool(call.name, call.arguments, config)
            logger.info(
                "Шаг %d/%d: инструмент %s, аргументы %s, результат %d символов",
                step,
                config.agent_max_steps,
                call.name,
                call.arguments,
                len(output),
            )
            messages.append(tool_message(call, output))

    return _finish_with_limit(partial_texts, config)
