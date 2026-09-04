"""Which run and which step is executing right now, without threading it through calls."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RunStep:
    """Coordinates of one loop iteration inside one run_agent call."""

    run_id: str
    step: int


@dataclass(frozen=True)
class PayloadMark:
    """How many messages the previous step of a run sent; the next step appends to them."""

    run_id: str
    message_count: int


_CURRENT: ContextVar[RunStep | None] = ContextVar("agent_run_step", default=None)
_PAYLOAD: ContextVar[PayloadMark | None] = ContextVar("agent_payload_mark", default=None)


def set_current_step(run_id: str, step: int) -> None:
    _CURRENT.set(RunStep(run_id=run_id, step=step))


def current_step() -> RunStep | None:
    """None outside the agent loop: stateless calls are not recorded."""
    return _CURRENT.get()


def remember_payload(run_id: str, message_count: int) -> None:
    _PAYLOAD.set(PayloadMark(run_id=run_id, message_count=message_count))


def previous_payload_size(run_id: str) -> int:
    """Messages already sent on the previous step of this run; 0 on the first step."""
    mark = _PAYLOAD.get()
    if mark is None or mark.run_id != run_id:
        return 0
    return mark.message_count
