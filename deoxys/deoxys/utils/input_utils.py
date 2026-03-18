import logging

import numpy as np

logger = logging.getLogger(__name__)


def input2action(device, controller_type="OSC_POSE", robot_name="Panda", gripper_dof=1):
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )

    drotation = raw_drotation[[1, 0, 2]]
    action = None

    if not reset:
        if controller_type == "OSC_POSE":
            drotation[2] = -drotation[2]
            drotation *= 75
            dpos *= 200

            grasp = 1 if grasp else -1
            action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

        if controller_type == "OSC_YAW":
            drotation[2] = -drotation[2]
            drotation *= 75
            dpos *= 200

            grasp = 1 if grasp else -1
            action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

        if controller_type == "OSC_POSITION":
            drotation[:] = 0
            dpos *= 200

            grasp = 1 if grasp else -1
            action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

        if controller_type == "JOINT_IMPEDANCE":
            grasp = 1 if grasp else -1
            action = np.array([0.0] * 7 + [grasp] * gripper_dof)

    return action, grasp


def input2actionInverted(device, controller_type="OSC_POSE", robot_name="Panda", gripper_dof=1):
    state = device.get_controller_state()
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )

    drotation = raw_drotation[[1, 0, 2]]

    dpos[[0, 1]] *= -1
    drotation[[0, 1]] *= -1
    drotation[2] = -drotation[2]

    dpos *= 200
    drotation *= 75

    grasp_val = 1 if grasp else -1

    if reset:
        return None, grasp_val

    if controller_type == "OSC_POSE":
        action = np.concatenate([dpos, drotation, [grasp_val] * gripper_dof])
    elif controller_type == "OSC_YAW":
        action = np.concatenate([dpos, drotation, [grasp_val] * gripper_dof])
    elif controller_type == "OSC_POSITION":
        drotation[:] = 0
        action = np.concatenate([dpos, drotation, [grasp_val] * gripper_dof])
    elif controller_type == "JOINT_IMPEDANCE":
        action = np.array([0.0] * 7 + [grasp_val] * gripper_dof)
    else:
        action = None

    return action, grasp_val


def inverseinput2action(device, controller_type="OSC_POSE", robot_name="Panda", gripper_dof=1):
    """Backward-compatible alias for the inverted SpaceMouse mapping."""

    return input2actionInverted(
        device=device,
        controller_type=controller_type,
        robot_name=robot_name,
        gripper_dof=gripper_dof,
    )
