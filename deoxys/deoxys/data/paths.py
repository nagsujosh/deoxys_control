"""Path helpers for canonical task/date/run layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def get_task_root(output_root: Path, task_name: str) -> Path:
    """Return the canonical root directory for one task."""

    return Path(output_root).expanduser().resolve() / task_name


def get_task_date_root(output_root: Path, task_name: str, date_str: str) -> Path:
    """Return the canonical root directory for one task/date pair."""

    return get_task_root(output_root, task_name) / date_str


def default_date_str() -> str:
    """Return the current date in YYYY-MM-DD format."""

    return datetime.now().strftime("%Y-%m-%d")


def next_run_dir(date_root: Path) -> Path:
    """Allocate the next available run directory under a task/date root."""

    date_root = Path(date_root)
    date_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in date_root.iterdir():
        if not child.is_dir() or not child.name.startswith("run"):
            continue
        suffix = child.name[3:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return date_root / f"run{max_index + 1}"


def is_date_dir(path: Path) -> bool:
    """Return whether one path name matches YYYY-MM-DD."""

    parts = path.name.split("-")
    return (
        len(parts) == 3
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and len(parts[2]) == 2
        and all(part.isdigit() for part in parts)
    )


def list_date_roots(task_root: Path) -> list[Path]:
    """Return all canonical date roots for one task."""

    task_root = Path(task_root)
    if not task_root.exists():
        return []
    return sorted(
        [path for path in task_root.iterdir() if path.is_dir() and is_date_dir(path)],
        key=lambda path: path.name,
    )


def allocate_run_dir(date_root: Path, max_attempts: int = 1000) -> Path:
    """Atomically allocate the next run directory under one date root.

    This is safer than computing `next_run_dir()` and creating it later because
    multiple operators may start collections for the same task/date at nearly
    the same time.
    """

    date_root = Path(date_root)
    date_root.mkdir(parents=True, exist_ok=True)
    for _ in range(max_attempts):
        candidate = next_run_dir(date_root)
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory under {date_root}")


def latest_date_root(task_root: Path) -> Optional[Path]:
    """Return the latest date directory for a task, if any exist."""

    candidates = list_date_roots(task_root)
    return candidates[-1] if candidates else None
