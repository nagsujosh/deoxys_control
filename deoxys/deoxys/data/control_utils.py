"""Shared control-path helpers for teleop, collection, replay, and reset."""

from __future__ import annotations

import time


def ensure_gripper_open(
    robot_interface,
    timeout_sec: float = 8.0,
    target_open_ratio: float = 0.98,
    command_period_sec: float = 0.1,
) -> bool:
    """Open the gripper before teleop/collection starts and verify it reached open.

    The Deoxys gripper API treats negative commands as "move open". We send an
    explicit open command and wait until the measured width is close to the
    reported maximum width so collection does not begin with a half-closed hand.
    If the gripper cannot be verified as open within the timeout, callers should
    treat that as a startup failure rather than silently proceeding.
    """

    if not getattr(robot_interface, "has_gripper", True):
        return True
    if not hasattr(robot_interface, "gripper_control"):
        return True

    start_sec = time.monotonic()
    last_command_sec = -float("inf")
    fallback_max_width = 0.08

    while True:
        max_width = fallback_max_width
        current_width = None
        if getattr(robot_interface, "gripper_state_buffer_size", 0) > 0:
            current_width_array = robot_interface.last_gripper_q
            if current_width_array is not None:
                current_width = float(current_width_array)
            latest_state = robot_interface._gripper_state_buffer[-1]
            max_width = float(getattr(latest_state, "max_width", fallback_max_width) or fallback_max_width)
            if current_width is not None and current_width >= max_width * target_open_ratio:
                return True

        now_sec = time.monotonic()
        if now_sec - last_command_sec >= command_period_sec:
            robot_interface.gripper_control(-1.0)
            last_command_sec = now_sec
        if now_sec - start_sec >= timeout_sec:
            return False
        time.sleep(0.05)


def termination_action_for_controller(controller_type: str) -> list[float]:
    """Return a controller-shaped neutral action used only for termination.

    The underlying Franka/Deoxys interface still expects an action payload even
    when `termination=True`. This helper keeps the payload shape consistent with
    the selected controller while avoiding controller-specific assertion errors.
    The gripper dimension is included for interface compatibility, but the
    interface skips gripper commands entirely during termination.
    """

    if controller_type in {"JOINT_POSITION", "JOINT_IMPEDANCE"}:
        return [0.0] * 7 + [-1.0]
    return [0.0] * 6 + [-1.0]
