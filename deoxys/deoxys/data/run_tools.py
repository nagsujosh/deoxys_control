"""Helpers for locating and describing raw teleoperation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import TaskConfig
from .paths import get_task_date_root, get_task_root, latest_date_root, list_date_roots


@dataclass(frozen=True)
class RunCameraSpec:
    """Camera layout recorded by one raw run manifest."""

    role: str
    camera_id: int
    enable_depth: bool


def list_run_dirs(date_root: Path) -> List[Path]:
    """Return run directories sorted by numeric run index."""

    runs = [
        path
        for path in Path(date_root).iterdir()
        if path.is_dir() and path.name.startswith("run") and path.name[3:].isdigit()
    ]
    return sorted(runs, key=lambda path: int(path.name[3:]))


def list_task_dates(task_cfg: TaskConfig) -> List[Dict[str, object]]:
    """Return all date roots for one task with basic artifact summary."""

    task_root = get_task_root(task_cfg.output_root, task_cfg.name)
    results: List[Dict[str, object]] = []
    for date_root in list_date_roots(task_root):
        results.append(
            {
                "date": date_root.name,
                "path": str(date_root.resolve()),
                "run_count": len(list_run_dirs(date_root)),
                "has_demo_hdf5": (date_root / "demo.hdf5").exists(),
            }
        )
    return results


def list_task_runs(
    task_cfg: TaskConfig,
    *,
    date_str: Optional[str] = None,
    all_dates: bool = False,
) -> List[Dict[str, object]]:
    """Return run summaries for one task, optionally across all dates."""

    task_root = get_task_root(task_cfg.output_root, task_cfg.name)
    if all_dates:
        date_roots = list_date_roots(task_root)
    elif date_str is None:
        latest = latest_date_root(task_root)
        if latest is None:
            return []
        date_roots = [latest]
    else:
        date_root = get_task_date_root(task_cfg.output_root, task_cfg.name, date_str)
        if not date_root.exists():
            return []
        date_roots = [date_root]

    results: List[Dict[str, object]] = []
    for date_root in date_roots:
        for run_dir in list_run_dirs(date_root):
            manifest_path = run_dir / "manifest.json"
            manifest = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}
            results.append(
                {
                    "date": date_root.name,
                    "run": run_dir.name,
                    "path": str(run_dir.resolve()),
                    "status": manifest.get("status", "unknown"),
                    "num_samples": int(manifest.get("num_samples", 0) or 0),
                }
            )
    return results


def resolve_run_dir(
    task_cfg: TaskConfig,
    date_str: Optional[str] = None,
    run_name: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> Path:
    """Resolve one raw run directory from task/date/run identifiers."""

    if run_dir:
        path = Path(run_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Run directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Expected a run directory path, got: {path}")
        return path

    if date_str is None:
        task_root = get_task_root(task_cfg.output_root, task_cfg.name)
        date_root = latest_date_root(task_root)
        if date_root is None:
            raise FileNotFoundError(f"No date directories found under {task_root}")
    else:
        date_root = get_task_date_root(task_cfg.output_root, task_cfg.name, date_str)
        if not date_root.exists():
            raise FileNotFoundError(f"Task/date root not found: {date_root}")

    run_dirs = list_run_dirs(date_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {date_root}")

    if run_name is None:
        return run_dirs[-1]

    normalized = run_name if run_name.startswith("run") else f"run{run_name}"
    path = date_root / normalized
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path.resolve()


def run_camera_specs(task_cfg: TaskConfig, manifest: Dict[str, object]) -> List[RunCameraSpec]:
    """Return the camera layout that should interpret one run.

    Prefer the camera roles/ids/depth flags captured in the run manifest so
    later task-config edits do not silently reinterpret previously recorded data.
    """

    manifest_roles = manifest.get("camera_roles")
    manifest_ids = manifest.get("camera_ids")
    manifest_depth = manifest.get("depth_enabled")
    if isinstance(manifest_roles, list) and isinstance(manifest_ids, dict):
        fallback_by_role = {camera.role: camera for camera in task_cfg.cameras}
        specs: List[RunCameraSpec] = []
        for role_value in manifest_roles:
            role = str(role_value)
            if role not in manifest_ids:
                continue
            fallback_camera = fallback_by_role.get(role)
            enable_depth = (
                bool(manifest_depth.get(role, False))
                if isinstance(manifest_depth, dict) and role in manifest_depth
                else (fallback_camera.enable_depth if fallback_camera is not None else False)
            )
            specs.append(
                RunCameraSpec(
                    role=role,
                    camera_id=int(manifest_ids[role]),
                    enable_depth=enable_depth,
                )
            )
        if specs:
            return specs

    return [
        RunCameraSpec(
            role=camera.role,
            camera_id=camera.camera_id,
            enable_depth=camera.enable_depth,
        )
        for camera in task_cfg.cameras
    ]
