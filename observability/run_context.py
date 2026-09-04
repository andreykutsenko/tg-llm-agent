"""Which run and which step is executing right now, without threading it through calls."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RunStep:
    """Coordinates of one loop iteration inside one run_agent call."""

    run_id: str
    step: int


_CURRENT: ContextVar[RunStep | None] = ContextVar("agent_run_step", default=None)


def set_current_step(run_id: str, step: int) -> None:
    _CURRENT.set(RunStep(run_id=run_id, step=step))


def current_step() -> RunStep | None:
    """None outside the agent loop: stateless calls are not recorded."""
    return _CURRENT.get()
