"""Plain SpaceMouse teleoperation using task-configured settings."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.input_utils import input2action, input2actionInverted
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.yaml_config import YamlConfig

from .config import ResetPreset, TaskConfig
from .control_utils import ensure_gripper_open, termination_action_for_controller
from .logging_utils import get_data_logger

logger = get_data_logger("teleop")


def _apply_action_transform(task_cfg: TaskConfig, action: np.ndarray) -> np.ndarray:
    """Apply configured per-axis action multipliers."""

    multipliers = np.asarray(task_cfg.collection.action_multipliers, dtype=np.float64)
    if multipliers.shape != action.shape:
        logger.warning(
            "Ignoring action multipliers with mismatched shape %s for action shape %s",
            multipliers.shape,
            action.shape,
        )
        return action
    return action * multipliers


def run_teleop(
    task_cfg: TaskConfig,
    preset: ResetPreset | None = None,
    max_steps: int = 0,
    inverse: bool = False,
) -> Dict[str, object]:
    """Run plain teleoperation without recording a dataset."""

    device = None
    robot_interface = None
    sent_commands = 0
    skipped_invalid_actions = 0
    status = "success"
    failure_reason = ""

    try:
        if preset is not None:
            from .reset import run_reset

            logger.info("Running reset preset `%s` before teleoperation", preset.name)
            try:
                run_reset(task_cfg, preset)
            except KeyboardInterrupt:
                logger.info("Stopping teleoperation because the reset preset was interrupted")
                status = "interrupted"
                return {
                    "task": task_cfg.name,
                    "status": status,
                    "preset": preset.name,
                    "controller_type": task_cfg.controller_type,
                    "sent_commands": sent_commands,
                    "skipped_invalid_actions": skipped_invalid_actions,
                    "failure_reason": "reset_interrupted",
                    "timestamp_unix_sec": time.time(),
                }
            except Exception as exc:
                logger.error("Teleoperation aborted because reset preset `%s` failed: %s", preset.name, exc)
                return {
                    "task": task_cfg.name,
                    "status": "failed",
                    "preset": preset.name,
                    "controller_type": task_cfg.controller_type,
                    "sent_commands": sent_commands,
                    "skipped_invalid_actions": skipped_invalid_actions,
                    "failure_reason": str(exc),
                    "timestamp_unix_sec": time.time(),
                }
        logger.info(
            "Starting teleoperation for task=%s controller=%s interface_cfg=%s policy_hz=%s camera_hz=%s",
            task_cfg.name,
            task_cfg.controller_type,
            task_cfg.interface_cfg_path,
            task_cfg.control_rates_hz.get("policy", 0),
            task_cfg.camera_capture_rates_hz,
        )
        device = SpaceMouse(
            vendor_id=task_cfg.spacemouse_vendor_id,
            product_id=task_cfg.spacemouse_product_id,
        )
        device.start_control()

        robot_interface = FrankaInterface(
            str(task_cfg.interface_cfg_path), use_visualizer=False
        )
        controller_cfg = YamlConfig(str(task_cfg.controller_cfg_path)).as_easydict()
        logger.info("Ensuring the gripper is fully open before teleoperation starts")
        if not ensure_gripper_open(robot_interface):
            failure_reason = "gripper_not_open"
            logger.error(
                "Aborting teleoperation because the gripper could not be verified fully open"
            )
            return {
                "task": task_cfg.name,
                "status": "failed",
                "preset": None if preset is None else preset.name,
                "controller_type": task_cfg.controller_type,
                "sent_commands": sent_commands,
                "skipped_invalid_actions": skipped_invalid_actions,
                "failure_reason": failure_reason,
                "timestamp_unix_sec": time.time(),
            }

        step = 0
        action_fn = input2actionInverted if inverse else input2action
        while max_steps <= 0 or step < max_steps:
            step += 1
            action, _ = action_fn(
                device=device,
                controller_type=task_cfg.controller_type,
            )
            if action is None:
                logger.info("Received SpaceMouse stop signal, stopping teleoperation cleanly")
                status = "stopped"
                break

            action = np.asarray(action, dtype=np.float64)
            if action.ndim != 1 or action.shape[0] < 7:
                skipped_invalid_actions += 1
                logger.warning("Skipping malformed SpaceMouse action with shape %s", action.shape)
                continue

            action = _apply_action_transform(task_cfg, action)
            if task_cfg.controller_type == "OSC_YAW":
                action[3:5] = 0.0
            elif task_cfg.controller_type == "OSC_POSITION":
                action[3:6] = 0.0

            robot_interface.control(
                controller_type=task_cfg.controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            sent_commands += 1
    except KeyboardInterrupt:
        logger.info("Stopping teleoperation on keyboard interrupt")
        status = "interrupted"
    except Exception as exc:
        logger.exception("Teleoperation failed: %s", exc)
        status = "failed"
        failure_reason = str(exc)
    finally:
        if robot_interface is not None:
            try:
                controller_cfg = YamlConfig(str(task_cfg.controller_cfg_path)).as_easydict()
                robot_interface.control(
                    controller_type=task_cfg.controller_type,
                    action=termination_action_for_controller(task_cfg.controller_type),
                    controller_cfg=controller_cfg,
                    termination=True,
                )
            except Exception as exc:
                logger.warning("Failed to send teleop termination command: %s", exc)
            finally:
                robot_interface.close()
                logger.info("Closed Franka interface")
        if device is not None:
            device.close()
            logger.info("Closed SpaceMouse device")

    return {
        "task": task_cfg.name,
        "status": status,
        "preset": None if preset is None else preset.name,
        "controller_type": task_cfg.controller_type,
        "sent_commands": sent_commands,
        "skipped_invalid_actions": skipped_invalid_actions,
        "failure_reason": failure_reason,
        "timestamp_unix_sec": time.time(),
    }
