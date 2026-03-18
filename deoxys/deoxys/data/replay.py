"""Robot replay for raw teleoperation runs."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.yaml_config import YamlConfig

from .config import ResetPreset, TaskConfig
from .control_utils import termination_action_for_controller
from .logging_utils import get_data_logger
from .run_tools import resolve_run_dir

logger = get_data_logger("replay")


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


def _policy_step_interval_sec(policy_rate_hz: int) -> float:
    """Return the paced replay interval in seconds for one control step."""

    return 0.0 if policy_rate_hz <= 0 else 1.0 / float(policy_rate_hz)


def _runtime_replay_settings(task_cfg: TaskConfig, manifest: Dict[str, object]) -> Dict[str, object]:
    """Resolve replay settings from the recorded run manifest when available."""

    manifest_rates = manifest.get("control_rates_hz", {})
    policy_rate_hz = task_cfg.control_rates_hz.get("policy", 0)
    if isinstance(manifest_rates, dict):
        policy_rate_hz = int(manifest_rates.get("policy", policy_rate_hz))

    return {
        "controller_type": str(manifest.get("controller_type", task_cfg.controller_type)),
        "interface_cfg_path": str(
            manifest.get("interface_cfg_path", str(task_cfg.interface_cfg_path))
        ),
        "controller_cfg_path": str(
            manifest.get("controller_cfg_path", str(task_cfg.controller_cfg_path))
        ),
        "policy_rate_hz": policy_rate_hz,
    }


def _task_cfg_with_manifest_interface(task_cfg: TaskConfig, manifest: Dict[str, object]) -> TaskConfig:
    """Build a task config that preserves the recorded robot interface path for reset/replay.

    Replay resets should target the same Franka interface config that the run was
    recorded with, even if the task YAML has been edited later.
    """

    interface_cfg = str(manifest.get("interface_cfg", task_cfg.interface_cfg))
    interface_cfg_path = str(manifest.get("interface_cfg_path", str(task_cfg.interface_cfg_path)))
    if interface_cfg_path.endswith(interface_cfg):
        return replace(task_cfg, interface_cfg=interface_cfg)

    # Fall back to the recorded relative interface config path if the manifest
    # does not preserve the short config stem separately.
    return replace(task_cfg, interface_cfg=interface_cfg_path)


def replay_run(
    task_cfg: TaskConfig,
    date_str: str | None = None,
    run_name: str | None = None,
    run_dir: str | None = None,
    preset: Optional[ResetPreset] = None,
    max_steps: int = 0,
) -> Dict[str, object]:
    """Replay one raw run on the robot using the canonical delta-action stream."""

    resolved_run_dir = resolve_run_dir(
        task_cfg=task_cfg, date_str=date_str, run_name=run_name, run_dir=run_dir
    )
    manifest_path = resolved_run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in (None, "success"):
        raise RuntimeError(
            f"Refusing to replay run `{resolved_run_dir}` because manifest status is "
            f"`{manifest.get('status')}`."
        )
    if manifest.get("action_semantics") not in (None, "delta_action"):
        raise RuntimeError(
            f"Refusing to replay run `{resolved_run_dir}` because it does not advertise "
            f"`delta_action` semantics."
        )

    action_path = _resolve_delta_action_path(resolved_run_dir, manifest)
    if action_path is None:
        raise FileNotFoundError(f"Replay delta-action file not found under: {resolved_run_dir}")
    actions = np.load(action_path)["data"]
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(
            f"Expected delta actions shaped [T, 7+] in {action_path}, got {actions.shape}"
        )
    replay_settings = _runtime_replay_settings(task_cfg, manifest)
    controller_type = str(replay_settings["controller_type"])
    interface_cfg_path = str(replay_settings["interface_cfg_path"])
    controller_cfg_path = str(replay_settings["controller_cfg_path"])
    policy_rate_hz = int(replay_settings["policy_rate_hz"])

    if preset is not None:
        from .reset import run_reset

        logger.info("Running reset preset `%s` before replay", preset.name)
        try:
            run_reset(_task_cfg_with_manifest_interface(task_cfg, manifest), preset)
        except KeyboardInterrupt:
            logger.info("Replay aborted because the reset preset was interrupted")
            return {
                "task": task_cfg.name,
                "run_dir": str(resolved_run_dir),
                "status": "interrupted",
                "controller_type": controller_type,
                "action_semantics": "delta_action",
                "replayed_steps": 0,
                "available_steps": int(actions.shape[0]),
                "nominal_policy_rate_hz": policy_rate_hz,
                "paced_replay_interval_sec": 0.0,
                "elapsed_sec": 0.0,
                "failure_reason": "reset_interrupted",
                "timestamp_unix_sec": time.time(),
            }
        except Exception as exc:
            logger.error("Replay aborted because reset preset `%s` failed: %s", preset.name, exc)
            return {
                "task": task_cfg.name,
                "run_dir": str(resolved_run_dir),
                "status": "failed",
                "controller_type": controller_type,
                "action_semantics": "delta_action",
                "replayed_steps": 0,
                "available_steps": int(actions.shape[0]),
                "nominal_policy_rate_hz": policy_rate_hz,
                "paced_replay_interval_sec": 0.0,
                "elapsed_sec": 0.0,
                "failure_reason": str(exc),
                "timestamp_unix_sec": time.time(),
            }

    step_interval_sec = _policy_step_interval_sec(policy_rate_hz)
    if step_interval_sec <= 0.0:
        logger.warning(
            "Replay pacing is disabled because the configured policy rate is %s Hz",
            policy_rate_hz,
        )

    replay_count = int(actions.shape[0] if max_steps <= 0 else min(actions.shape[0], max_steps))
    logger.info(
        "Replaying %s delta-action steps from %s with controller=%s nominal_policy_hz=%s",
        replay_count,
        resolved_run_dir,
        controller_type,
        policy_rate_hz,
    )

    robot_interface = None
    controller_cfg = None
    sent_commands = 0
    status = "success"
    start_time_sec = None
    failure_reason = ""

    try:
        controller_cfg = YamlConfig(controller_cfg_path).as_easydict()
        if not getattr(controller_cfg, "is_delta", False):
            logger.warning(
                "Replaying delta-action data through controller `%s`, which is not marked "
                "delta-based in the controller config. Replay may not match the original run.",
                controller_type,
            )
        robot_interface = FrankaInterface(
            interface_cfg_path,
            use_visualizer=False,
        )
        start_time_sec = time.time()
        start_monotonic_sec = time.monotonic()
        for index in range(replay_count):
            action = np.asarray(actions[index], dtype=np.float64)
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            sent_commands += 1
            if step_interval_sec > 0.0:
                next_deadline_sec = start_monotonic_sec + (index + 1) * step_interval_sec
                remaining_sec = next_deadline_sec - time.monotonic()
                if remaining_sec > 0.0:
                    time.sleep(remaining_sec)
            if sent_commands == 1 or sent_commands % 100 == 0:
                logger.info("Replayed %s/%s delta-action steps", sent_commands, replay_count)
        elapsed_sec = time.time() - start_time_sec
    except KeyboardInterrupt:
        status = "interrupted"
        elapsed_sec = 0.0 if start_time_sec is None else time.time() - start_time_sec
        logger.info("Stopped replay on keyboard interrupt")
    except Exception as exc:
        status = "failed"
        elapsed_sec = 0.0 if start_time_sec is None else time.time() - start_time_sec
        failure_reason = str(exc)
        logger.exception("Replay failed: %s", exc)
    finally:
        if robot_interface is not None:
            try:
                if controller_cfg is not None:
                    robot_interface.control(
                        controller_type=controller_type,
                        action=termination_action_for_controller(controller_type),
                        controller_cfg=controller_cfg,
                        termination=True,
                    )
            except Exception as exc:
                logger.warning("Failed to send replay termination command: %s", exc)
            finally:
                robot_interface.close()
                logger.info("Closed Franka interface after replay")

    return {
        "task": task_cfg.name,
        "run_dir": str(resolved_run_dir),
        "status": status,
        "controller_type": controller_type,
        "action_semantics": "delta_action",
        "replayed_steps": sent_commands,
        "available_steps": int(actions.shape[0]),
        "nominal_policy_rate_hz": policy_rate_hz,
        "paced_replay_interval_sec": step_interval_sec,
        "elapsed_sec": elapsed_sec,
        "failure_reason": failure_reason,
        "timestamp_unix_sec": time.time(),
    }
