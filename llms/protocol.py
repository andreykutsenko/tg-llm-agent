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
class LLMResult:
    """Either a final text answer or a list of tool calls the model asks for."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

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
