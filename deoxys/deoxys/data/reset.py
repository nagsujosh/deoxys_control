"""Named reset preset execution."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.yaml_config import YamlConfig

from .config import ResetPreset, TaskConfig
from .control_utils import termination_action_for_controller
from .logging_utils import get_data_logger

logger = get_data_logger("reset")


def _preset_joint_positions(preset: ResetPreset) -> np.ndarray:
    joints = np.array(preset.joint_positions, dtype=np.float64)
    if preset.allow_jitter and preset.jitter_std_rad > 0.0:
        jitter = np.random.randn(len(joints)) * preset.jitter_std_rad
        if preset.jitter_clip_rad > 0.0:
            jitter = np.clip(jitter, -preset.jitter_clip_rad, preset.jitter_clip_rad)
        joints = joints + jitter
    return joints


def run_reset(task_cfg: TaskConfig, preset: ResetPreset) -> Dict[str, object]:
    """Run a named reset preset to convergence."""

    logger.info("Running reset preset `%s`", preset.name)
    controller_cfg = YamlConfig(str(preset.controller_cfg_path)).as_easydict()
    robot_interface = FrankaInterface(
        str(task_cfg.interface_cfg_path), use_visualizer=False
    )

    target_joints = _preset_joint_positions(preset)
    action = target_joints.tolist() + [-1.0]
    status = "success"
    start_time = np.float64(time.monotonic())

    try:
        while True:
            elapsed_sec = float(time.monotonic() - start_time)
            if elapsed_sec > preset.timeout_sec:
                raise TimeoutError(
                    f"Reset preset `{preset.name}` did not converge within {preset.timeout_sec:.2f} seconds."
                )
            if robot_interface.state_buffer_size > 0:
                error = np.max(np.abs(np.array(robot_interface.last_q) - target_joints))
                if error < preset.tolerance_rad:
                    break
            robot_interface.control(
                controller_type=preset.controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
    except KeyboardInterrupt:
        status = "interrupted"
        logger.info("Reset preset `%s` interrupted", preset.name)
        raise
    except Exception:
        status = "failed"
        logger.exception("Reset preset `%s` failed", preset.name)
        raise
    finally:
        try:
            robot_interface.control(
                controller_type=preset.controller_type,
                action=termination_action_for_controller(preset.controller_type),
                controller_cfg=controller_cfg,
                termination=True,
            )
        except Exception as exc:
            logger.warning("Failed to send reset termination command: %s", exc)
        finally:
            robot_interface.close()
        if status == "success":
            logger.info("Reset preset `%s` completed", preset.name)

    return {
        "preset": preset.name,
        "status": status,
        "target_joint_positions_rad": target_joints.tolist(),
        "tolerance_rad": preset.tolerance_rad,
        "timeout_sec": preset.timeout_sec,
        "allow_jitter": preset.allow_jitter,
    }
