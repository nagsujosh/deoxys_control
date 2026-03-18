"""Validation, replay, and summary tooling for raw teleoperation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zipfile import BadZipFile

import numpy as np

from .config import TaskConfig
from .logging_utils import get_data_logger
from .run_tools import resolve_run_dir, run_camera_specs
from .viewer import _load_cv2, _normalize_depth, _overlay_status, _stack_panels_vertically

logger = get_data_logger("validate")


def _resolve_delta_action_path(run_dir: Path, manifest: Dict[str, object]) -> Path | None:
    """Resolve the canonical delta-action file, with guarded legacy fallback."""

    canonical = run_dir / "testing_demo_delta_action.npz"
    if canonical.exists():
        return canonical

    legacy = run_dir / "testing_demo_action.npz"
    if legacy.exists() and manifest.get("action_semantics") == "delta_action":
        logger.warning(
            "Using legacy delta-action file %s because manifest still advertises delta_action semantics",
            legacy,
        )
        return legacy
    return None


def _load_manifest(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _npz_length(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if "data" in data:
                return int(data["data"].shape[0])
            if data.files:
                return int(data[data.files[0]].shape[0])
            return 0
    except (BadZipFile, EOFError, OSError, ValueError, RuntimeError, KeyError, Exception):
        return None


def _npz_internal_lengths(path: Path) -> Dict[str, int]:
    """Return the first-dimension length of every array stored in one NPZ file."""

    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as data:
            lengths: Dict[str, int] = {}
            for name in data.files:
                array = np.asarray(data[name])
                lengths[name] = int(array.shape[0]) if array.ndim > 0 else 1
            return lengths
    except (BadZipFile, EOFError, OSError, ValueError, RuntimeError, KeyError, Exception):
        return {}


def _npz_corruption_issue(path: Path) -> str | None:
    """Return a human-readable corruption issue if an NPZ file cannot be read."""

    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            # Force archive decompression eagerly so partial/corrupt files fail here.
            for name in data.files:
                np.asarray(data[name])
    except (BadZipFile, EOFError, OSError, ValueError, RuntimeError, KeyError, Exception) as exc:
        return f"{path.name} is unreadable/corrupt: {exc}"
    return None


def _timing_stats(values: np.ndarray) -> Dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite_count": 0, "mean": None, "max_abs": None}
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "max_abs": float(np.max(np.abs(finite))),
    }


def _contiguous_segments(mask: np.ndarray) -> Tuple[int, int]:
    count = 0
    max_len = 0
    current = 0
    for flagged in mask.astype(bool):
        if flagged:
            current += 1
            max_len = max(max_len, current)
        elif current > 0:
            count += 1
            current = 0
    if current > 0:
        count += 1
    return count, max_len


def _zero_segment_summary(values: np.ndarray, threshold: float) -> Dict[str, int]:
    flattened = values.reshape(values.shape[0], -1)
    suspicious = np.sum(np.abs(flattened), axis=1) <= threshold
    count, max_len = _contiguous_segments(suspicious)
    return {
        "num_suspicious_samples": int(np.sum(suspicious)),
        "num_segments": count,
        "max_segment_length": max_len,
    }


def _required_run_files(task_cfg: TaskConfig, manifest: Dict[str, object]) -> List[str]:
    files = [
        "manifest.json",
        "testing_demo_ee_state_10d.npz",
    ]
    for camera_spec in run_camera_specs(task_cfg, manifest):
        camera_id = camera_spec.camera_id
        files.extend(
            [
                f"testing_demo_camera_{camera_id}_color.npz",
                f"testing_demo_camera_{camera_id}_valid.npz",
                f"testing_demo_camera_{camera_id}_calibration.json",
            ]
        )
        if camera_spec.enable_depth:
            files.append(f"testing_demo_camera_{camera_id}_depth.npz")
            files.append(f"testing_demo_camera_{camera_id}_depth_valid.npz")
    return files


def _camera_panel(
    role: str,
    color: np.ndarray,
    valid: int,
    frame_idx: int,
    depth_state: str = "disabled",
) -> np.ndarray:
    cv2 = _load_cv2()
    rgb = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return _overlay_status(
        rgb,
        [
            f"role: {role}",
            f"sample: {frame_idx}",
            f"valid: {'yes' if int(valid) > 0 else 'placeholder'}",
            f"depth: {depth_state}",
        ],
    )


def _play_run(task_cfg: TaskConfig, run_dir: Path, manifest: Dict[str, object], fps: float, include_depth: bool) -> None:
    cv2 = _load_cv2()
    colors: Dict[str, np.ndarray] = {}
    depths: Dict[str, np.ndarray] = {}
    valids: Dict[str, np.ndarray] = {}
    depth_valids: Dict[str, np.ndarray] = {}

    frame_count = None
    run_cameras = run_camera_specs(task_cfg, manifest)
    for camera_cfg in run_cameras:
        color_path = run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_color.npz"
        if not color_path.exists():
            continue
        colors[camera_cfg.role] = np.load(color_path)["data"]
        frame_count = colors[camera_cfg.role].shape[0] if frame_count is None else min(
            frame_count, colors[camera_cfg.role].shape[0]
        )

        valid_path = run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_valid.npz"
        if valid_path.exists():
            valids[camera_cfg.role] = np.load(valid_path)["data"].reshape(-1)
        else:
            valids[camera_cfg.role] = np.ones(colors[camera_cfg.role].shape[0], dtype=np.uint8)

        depth_path = run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_depth.npz"
        depth_valid_path = run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_depth_valid.npz"
        if depth_valid_path.exists():
            depth_valids[camera_cfg.role] = np.load(depth_valid_path)["data"].reshape(-1)
        elif camera_cfg.enable_depth:
            depth_valids[camera_cfg.role] = np.ones(colors[camera_cfg.role].shape[0], dtype=np.uint8)

        if include_depth and depth_path.exists():
            depths[camera_cfg.role] = np.load(depth_path)["data"]

    if frame_count is None:
        raise FileNotFoundError(f"No camera color arrays found in {run_dir}")

    delay_ms = max(1, int(1000.0 / max(fps, 1.0)))
    logger.info("Replaying %s frames from %s", frame_count, run_dir)

    while True:
        for frame_idx in range(frame_count):
            panels: List[np.ndarray] = []
            for camera_cfg in run_cameras:
                role = camera_cfg.role
                if role not in colors:
                    continue
                depth_state = "disabled"
                if camera_cfg.enable_depth:
                    if role in depth_valids and int(depth_valids[role][frame_idx]) > 0:
                        depth_state = "valid"
                    elif role in depths:
                        depth_state = "placeholder"
                    else:
                        depth_state = "unavailable"
                panel = _camera_panel(
                    role,
                    colors[role][frame_idx],
                    valids[role][frame_idx],
                    frame_idx,
                    depth_state=depth_state,
                )
                if (
                    include_depth
                    and role in depths
                    and int(depth_valids.get(role, np.ones(1, dtype=np.uint8))[frame_idx]) > 0
                ):
                    depth_vis = _normalize_depth(depths[role][frame_idx])
                    if depth_vis is not None:
                        panel = np.hstack([panel, depth_vis])
                panels.append(panel)

            if not panels:
                break
            canvas = _stack_panels_vertically(panels)
            cv2.imshow("deoxys.data validate", canvas)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                logger.info("Validation replay closed")
                return
        break
    cv2.destroyAllWindows()


def validate_run(
    task_cfg: TaskConfig,
    date_str: str | None = None,
    run_name: str | None = None,
    run_dir: str | None = None,
    play: bool = False,
    fps: float = 10.0,
    include_depth: bool = False,
) -> Dict[str, object]:
    """Validate a raw run and return a structured summary."""

    resolved_run_dir = resolve_run_dir(
        task_cfg=task_cfg, date_str=date_str, run_name=run_name, run_dir=run_dir
    )
    manifest = _load_manifest(resolved_run_dir)
    delta_action_path = _resolve_delta_action_path(resolved_run_dir, manifest)
    missing_files = [
        name
        for name in _required_run_files(task_cfg, manifest)
        if not (resolved_run_dir / name).exists()
    ]
    if delta_action_path is None:
        missing_files.append("testing_demo_delta_action.npz")
    corrupt_files: List[str] = []
    archive_paths = [
        resolved_run_dir / name
        for name in _required_run_files(task_cfg, manifest)
        if (resolved_run_dir / name).exists() and name.endswith(".npz")
    ]
    if delta_action_path is not None:
        archive_paths.append(delta_action_path)
    seen_paths: set[Path] = set()
    for archive_path in archive_paths:
        if archive_path in seen_paths:
            continue
        seen_paths.add(archive_path)
        corruption_issue = _npz_corruption_issue(archive_path)
        if corruption_issue is not None:
            corrupt_files.append(corruption_issue)

    stream_files = {
        "delta_actions": None if delta_action_path is None else delta_action_path.name,
        "ee_state_10d": "testing_demo_ee_state_10d.npz",
    }
    for name, filename in (
        ("obs_delta_eef", "testing_demo_observed_delta_eef.npz"),
        ("obs_delta_joint_states", "testing_demo_observed_delta_joint_states.npz"),
        ("pose_tracking_errors", "testing_demo_pose_tracking_error.npz"),
        ("joint_tracking_errors", "testing_demo_joint_tracking_error.npz"),
        ("gripper_tracking_errors", "testing_demo_gripper_tracking_error.npz"),
        ("ee_states", "testing_demo_ee_states.npz"),
        ("joint_states", "testing_demo_joint_states.npz"),
        ("gripper_states", "testing_demo_gripper_states.npz"),
        ("sync_metadata", "testing_demo_sync_metadata.npz"),
        ("robot_metadata", "testing_demo_robot_metadata.npz"),
    ):
        if (resolved_run_dir / filename).exists():
            stream_files[name] = filename
    run_cameras = run_camera_specs(task_cfg, manifest)
    for camera_cfg in run_cameras:
        stream_files[f"{camera_cfg.role}_rgb"] = f"testing_demo_camera_{camera_cfg.camera_id}_color.npz"
        stream_files[f"{camera_cfg.role}_valid"] = (
            f"testing_demo_camera_{camera_cfg.camera_id}_valid.npz"
        )
        stream_files[f"{camera_cfg.role}_info"] = (
            f"testing_demo_camera_{camera_cfg.camera_id}_info.npz"
        )
        if camera_cfg.enable_depth:
            stream_files[f"{camera_cfg.role}_depth"] = (
                f"testing_demo_camera_{camera_cfg.camera_id}_depth.npz"
            )
            stream_files[f"{camera_cfg.role}_depth_valid"] = (
                f"testing_demo_camera_{camera_cfg.camera_id}_depth_valid.npz"
            )

    stream_lengths = {
        name: (None if filename is None else _npz_length(resolved_run_dir / filename))
        for name, filename in stream_files.items()
    }
    present_lengths = {name: length for name, length in stream_lengths.items() if length is not None}
    npz_internal_issues: List[str] = []
    for archive_name in ("testing_demo_sync_metadata.npz", "testing_demo_robot_metadata.npz"):
        archive_path = resolved_run_dir / archive_name
        internal_lengths = _npz_internal_lengths(archive_path)
        if internal_lengths and len(set(internal_lengths.values())) != 1:
            detail = ", ".join(
                f"{field}={length}" for field, length in sorted(internal_lengths.items())
            )
            npz_internal_issues.append(f"{archive_name} has internally mismatched field lengths: {detail}")

    sync_stats: Dict[str, Dict[str, float | None]] = {}
    sync_path = resolved_run_dir / "testing_demo_sync_metadata.npz"
    if sync_path.exists():
        corruption_issue = _npz_corruption_issue(sync_path)
        if corruption_issue is None:
            with np.load(sync_path) as sync_data:
                for field_name in (
                    "robot_state_age_sec",
                    "gripper_state_age_sec",
                    "agentview_frame_age_sec",
                    "agentview_robot_receive_skew_sec",
                    "wrist_frame_age_sec",
                    "wrist_robot_receive_skew_sec",
                ):
                    if field_name in sync_data:
                        sync_stats[field_name] = _timing_stats(
                            np.asarray(sync_data[field_name]).reshape(-1)
                        )

    camera_stats: Dict[str, Dict[str, int]] = {}
    for camera_cfg in run_cameras:
        valid_path = resolved_run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_valid.npz"
        depth_valid_path = resolved_run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_depth_valid.npz"
        info_path = resolved_run_dir / f"testing_demo_camera_{camera_cfg.camera_id}_info.npz"
        valid_count = 0
        missing_count = 0
        missing_depth_count = 0
        frame_jump_count = 0
        if valid_path.exists() and _npz_corruption_issue(valid_path) is None:
            with np.load(valid_path) as valid_data:
                valid = valid_data["data"].reshape(-1)
                valid_count = int(np.sum(valid > 0))
                missing_count = int(np.sum(valid <= 0))
        if depth_valid_path.exists() and _npz_corruption_issue(depth_valid_path) is None:
            with np.load(depth_valid_path) as depth_valid_data:
                depth_valid = depth_valid_data["data"].reshape(-1)
                missing_depth_count = int(np.sum(depth_valid <= 0))
        if info_path.exists() and _npz_corruption_issue(info_path) is None:
            with np.load(info_path, allow_pickle=True) as info_data:
                info = info_data["data"]
                frame_ids = np.array(
                    [entry.get("frame_id", -1) for entry in info if entry.get("frame_id", -1) >= 0],
                    dtype=np.int64,
                )
                if frame_ids.size >= 2:
                    frame_jump_count = int(np.sum(np.diff(frame_ids) > 1))
        camera_stats[camera_cfg.role] = {
            "valid_frames": valid_count,
            "missing_or_placeholder_frames": missing_count,
            "missing_or_placeholder_depth_frames": missing_depth_count,
            "frame_jump_count": frame_jump_count,
        }

    zero_state_segments: Dict[str, Dict[str, int]] = {}
    threshold = float(task_cfg.collection.state_zero_threshold)
    for field_name, filename in (
        ("ee_states", "testing_demo_ee_states.npz"),
        ("joint_states", "testing_demo_joint_states.npz"),
        ("gripper_states", "testing_demo_gripper_states.npz"),
    ):
        path = resolved_run_dir / filename
        if path.exists() and _npz_corruption_issue(path) is None:
            with np.load(path) as values_data:
                values = values_data["data"]
            zero_state_segments[field_name] = _zero_segment_summary(values, threshold)

    issues: List[str] = []
    if manifest.get("status") != "success":
        issues.append(f"manifest status is {manifest.get('status', 'missing')}")
    if missing_files:
        issues.append(f"missing files: {', '.join(sorted(missing_files))}")
    issues.extend(sorted(corrupt_files))
    if present_lengths and len(set(present_lengths.values())) != 1:
        issues.append("stream lengths are mismatched")
    issues.extend(npz_internal_issues)
    for field_name, summary in zero_state_segments.items():
        if summary["num_segments"] > 0:
            issues.append(
                f"{field_name} contains {summary['num_segments']} suspicious near-zero segment(s)"
            )

    report = {
        "ok": len(issues) == 0,
        "run_dir": str(resolved_run_dir),
        "manifest_status": manifest.get("status"),
        "num_samples_manifest": manifest.get("num_samples"),
        "configured_rates_hz": {
            "control": (
                manifest.get("control_rates_hz")
                if isinstance(manifest.get("control_rates_hz"), dict)
                else task_cfg.control_rates_hz
            ),
            "camera_capture": (
                manifest.get("camera_capture_rates_hz")
                if isinstance(manifest.get("camera_capture_rates_hz"), dict)
                else task_cfg.camera_capture_rates_hz
            ),
        },
        "stream_lengths": stream_lengths,
        "timing_stats_sec": sync_stats,
        "camera_stats": camera_stats,
        "zero_state_segments": zero_state_segments,
        "skip_counters": manifest.get("skip_counters", {}),
        "issues": issues,
    }
    if play:
        if not report["ok"]:
            raise RuntimeError(
                "Refusing to replay invalid run during validation because structural issues were detected: "
                + "; ".join(issues)
            )
        _play_run(task_cfg, resolved_run_dir, manifest, fps=fps, include_depth=include_depth)
    return report
