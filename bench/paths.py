"""Where a measurement lives: the aggregate is committed, the raw JSONL is not."""

from pathlib import Path

from config import PROJECT_ROOT

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
DEFAULT_TELEMETRY_DIR = PROJECT_ROOT / "telemetry"


def measurement_path(results_dir: Path, label: str) -> Path:
    """Aggregated result of one measurement — goes into git."""
    return results_dir / f"{label}.json"


def outcomes_path(telemetry_dir: Path, label: str) -> Path:
    """Raw per-task verdicts with answer excerpts — stays local."""
    return telemetry_dir / f"{label}.outcomes.jsonl"


def events_path(telemetry_dir: Path, label: str) -> Path:
    """Raw telemetry events of the measurement — stays local."""
    return telemetry_dir / f"{label}.events.jsonl"
