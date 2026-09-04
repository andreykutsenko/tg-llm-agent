"""Provider-independent shape of a model step: text, tool calls, messages."""

from dataclasses import dataclass, field

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# Провайдер отдаёт аргументы строкой JSON; неразобранная строка доезжает
# до инструмента под этим ключом и превращается в понятный модели отказ.
INVALID_ARGUMENTS_KEY = "_invalid_json"


@dataclass(frozen=True)
class ToolSpec:
    """What a tool is called, what it does and which arguments it accepts."""

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run one tool."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Usage:
    """Token accounting of one model call; a provider without data reports zeros."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: int = 0
    # Запись в кэш Anthropic тарифицируется дороже обычного ввода (×1.25),
    # поэтому без этого поля стоимость замера «после» была бы занижена.
    cache_write_input_tokens: int = 0
    # Токены рассуждения, если провайдер отдаёт их отдельно; у Anthropic они
    # уже входят в output_tokens, поэтому здесь 0.
    reasoning_tokens: int = 0

    @property
    def is_reported(self) -> bool:
        return bool(self.input_tokens or self.output_tokens)


@dataclass(frozen=True)
class LLMResult:
    """Either a final text answer or a list of tool calls the model asks for."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def user_message(text: str) -> dict:
    return {"role": ROLE_USER, "content": text}


def assistant_message(result: LLMResult) -> dict:
    return {
        "role": ROLE_ASSISTANT,
        "content": result.text,
        "tool_calls": list(result.tool_calls),
    }


def tool_message(call: ToolCall, content: str) -> dict:
    return {
        "role": ROLE_TOOL,
        "tool_call_id": call.id,
        "name": call.name,
        "content": content,
    }
