"""Shared field definitions for raw runs and HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class FieldSpec:
    """Describes one numeric field and its physical convention."""

    name: str
    units: str
    description: str
    frame: str = ""


CORE_DATASET_SPECS: Dict[str, FieldSpec] = {
    "delta_actions": FieldSpec(
        name="delta_actions",
        units="controller-native delta action; translation delta in meters, rotation delta in radians, gripper delta unitless",
        description="Canonical teleoperation delta action stream sent to the controller for each accepted sample.",
        frame="controller command frame",
    ),
    "ee_states": FieldSpec(
        name="ee_states",
        units="4x4 homogeneous transform; translation in meters, rotation as unitless rotation matrix",
        description="Measured end-effector world pose from Deoxys `O_T_EE`.",
        frame="world",
    ),
    "ee_state_10d": FieldSpec(
        name="ee_state_10d",
        units="position in meters, unitless 6D rotation representation, then gripper width in meters",
        description="Measured learning-facing end-effector state represented as xyz position, 6D rotation, and gripper width.",
        frame="world plus gripper jaws",
    ),
    "joint_states": FieldSpec(
        name="joint_states",
        units="radians",
        description="Measured robot joint positions.",
        frame="robot joints",
    ),
    "gripper_states": FieldSpec(
        name="gripper_states",
        units="meters",
        description="Measured gripper width.",
        frame="gripper jaws",
    ),
    "obs_delta_eef": FieldSpec(
        name="obs_delta_eef",
        units="translation delta in meters, rotation delta in axis-angle radians",
        description="Observed end-effector delta between consecutive accepted samples.",
        frame="world",
    ),
    "obs_delta_joint_states": FieldSpec(
        name="obs_delta_joint_states",
        units="radians",
        description="Observed joint-position delta between consecutive accepted samples.",
        frame="robot joints",
    ),
    "pose_tracking_errors": FieldSpec(
        name="pose_tracking_errors",
        units="translation error in meters, rotation error in axis-angle radians",
        description="Measured end-effector tracking error relative to Deoxys desired pose `O_T_EE_d`.",
        frame="world",
    ),
    "joint_tracking_errors": FieldSpec(
        name="joint_tracking_errors",
        units="radians",
        description="Measured joint-position tracking error computed as q - q_d.",
        frame="robot joints",
    ),
    "gripper_tracking_errors": FieldSpec(
        name="gripper_tracking_errors",
        units="meters",
        description="Approximate gripper width tracking error computed from the teleoperation open/close command target width.",
        frame="gripper jaws",
    ),
    "agentview_rgb": FieldSpec(
        name="agentview_rgb",
        units="uint8 RGB image",
        description="Color stream from the external agent-view camera.",
        frame="camera optical frame after resize",
    ),
    "eye_in_hand_rgb": FieldSpec(
        name="eye_in_hand_rgb",
        units="uint8 RGB image",
        description="Color stream from the wrist-mounted camera.",
        frame="camera optical frame after resize",
    ),
    "agentview_depth": FieldSpec(
        name="agentview_depth",
        units="raw depth units declared in dataset attrs",
        description="Depth stream from the external agent-view camera.",
        frame="camera optical frame after resize",
    ),
    "eye_in_hand_depth": FieldSpec(
        name="eye_in_hand_depth",
        units="raw depth units declared in dataset attrs",
        description="Depth stream from the wrist-mounted camera.",
        frame="camera optical frame after resize",
    ),
}


ROBOT_METADATA_SPECS: Dict[str, FieldSpec] = {
    "O_T_EE_d": FieldSpec(
        "O_T_EE_d",
        "4x4 homogeneous transform; translation in meters, rotation as unitless rotation matrix",
        "Desired end-effector world pose from Franka/Deoxys.",
        "world",
    ),
    "O_T_EE_c": FieldSpec(
        "O_T_EE_c",
        "4x4 homogeneous transform; translation in meters, rotation as unitless rotation matrix",
        "Commanded end-effector world pose from Franka/Deoxys.",
        "world",
    ),
    "q_d": FieldSpec("q_d", "radians", "Desired robot joint positions.", "robot joints"),
    "dq": FieldSpec("dq", "radians/second", "Measured robot joint velocities.", "robot joints"),
    "dq_d": FieldSpec("dq_d", "radians/second", "Desired robot joint velocities.", "robot joints"),
    "ddq_d": FieldSpec("ddq_d", "radians/second^2", "Desired robot joint accelerations.", "robot joints"),
    "tau_J": FieldSpec("tau_J", "newton-meter", "Measured joint torques.", "robot joints"),
    "tau_J_d": FieldSpec("tau_J_d", "newton-meter", "Desired joint torques.", "robot joints"),
    "dtau_J": FieldSpec("dtau_J", "newton-meter/second", "Joint torque derivative estimate.", "robot joints"),
    "tau_ext_hat_filtered": FieldSpec(
        "tau_ext_hat_filtered",
        "newton-meter",
        "Estimated external joint torques.",
        "robot joints",
    ),
    "O_F_ext_hat_K": FieldSpec(
        "O_F_ext_hat_K",
        "Franka wrench units",
        "Estimated external wrench in world frame.",
        "world",
    ),
    "K_F_ext_hat_K": FieldSpec(
        "K_F_ext_hat_K",
        "Franka wrench units",
        "Estimated external wrench in stiffness frame.",
        "stiffness frame",
    ),
    "robot_mode": FieldSpec(
        "robot_mode",
        "enum",
        "Franka robot mode reported by Deoxys.",
        "robot status",
    ),
    "control_command_success_rate": FieldSpec(
        "control_command_success_rate",
        "ratio",
        "Controller-side command success rate reported by Deoxys.",
        "robot status",
    ),
    "frame": FieldSpec(
        "frame",
        "frame index",
        "Monotonic robot state frame identifier.",
        "robot status",
    ),
    "robot_time_sec": FieldSpec(
        "robot_time_sec",
        "seconds",
        "Robot-side timestamp in seconds.",
        "robot status",
    ),
    "gripper_max_width": FieldSpec(
        "gripper_max_width",
        "meters",
        "Maximum gripper width.",
        "gripper jaws",
    ),
    "gripper_is_grasped": FieldSpec(
        "gripper_is_grasped",
        "bool",
        "Whether the gripper reports a successful grasp.",
        "gripper status",
    ),
    "gripper_temperature": FieldSpec(
        "gripper_temperature",
        "celsius",
        "Reported gripper temperature.",
        "gripper status",
    ),
    "gripper_time_sec": FieldSpec(
        "gripper_time_sec",
        "seconds",
        "Gripper-side timestamp in seconds.",
        "gripper status",
    ),
    "current_errors": FieldSpec(
        "current_errors",
        "binary indicator vector",
        "Current Franka error flags ordered according to the Deoxys error proto.",
        "robot status",
    ),
    "last_motion_errors": FieldSpec(
        "last_motion_errors",
        "binary indicator vector",
        "Last-motion Franka error flags ordered according to the Deoxys error proto.",
        "robot status",
    ),
}


CAMERA_METADATA_SPECS: Dict[str, FieldSpec] = {
    "valid": FieldSpec(
        "valid",
        "uint8 mask where 1=real frame and 0=placeholder",
        "Per-sample indicator that marks whether the stored camera frame came from a real Redis update or from an optional-camera placeholder inserted to preserve alignment.",
    ),
    "depth_valid": FieldSpec(
        "depth_valid",
        "uint8 mask where 1=real depth frame and 0=placeholder or unavailable depth",
        "Per-sample indicator that marks whether the stored depth image came from a real depth frame or from a zero-filled placeholder used to preserve alignment.",
    ),
    "frame_id": FieldSpec("frame_id", "frame index", "Source camera frame identifier."),
    "acquisition_timestamp_ms": FieldSpec(
        "acquisition_timestamp_ms",
        "milliseconds",
        "Camera acquisition timestamp from the driver.",
    ),
    "publish_timestamp_sec": FieldSpec(
        "publish_timestamp_sec",
        "seconds since unix epoch",
        "Wall-clock publish timestamp recorded before Redis transport.",
    ),
    "frame_age_sec": FieldSpec(
        "frame_age_sec",
        "seconds",
        "Age of the camera frame at sample acceptance time using host wall-clock time.",
    ),
    "robot_receive_skew_sec": FieldSpec(
        "robot_receive_skew_sec",
        "seconds",
        "Host-clock skew between the camera publish timestamp and the received robot-state timestamp used for the same sample.",
    ),
    "depth_scale_m_per_unit": FieldSpec(
        "depth_scale_m_per_unit",
        "meters/unit",
        "Depth conversion factor from stored raw values to meters.",
    ),
    "sample_host_time_sec": FieldSpec(
        "sample_host_time_sec",
        "seconds since unix epoch",
        "Host wall-clock time when the collector accepted the sample.",
    ),
    "robot_state_receive_timestamp_sec": FieldSpec(
        "robot_state_receive_timestamp_sec",
        "seconds since unix epoch",
        "Host wall-clock time when the robot state used for the sample was received over ZMQ.",
    ),
    "robot_state_age_sec": FieldSpec(
        "robot_state_age_sec",
        "seconds",
        "Age of the robot state at sample acceptance time using host wall-clock time.",
    ),
    "gripper_state_receive_timestamp_sec": FieldSpec(
        "gripper_state_receive_timestamp_sec",
        "seconds since unix epoch",
        "Host wall-clock time when the gripper state used for the sample was received over ZMQ.",
    ),
    "gripper_state_age_sec": FieldSpec(
        "gripper_state_age_sec",
        "seconds",
        "Age of the gripper state at sample acceptance time using host wall-clock time.",
    ),
}


PROMOTED_METADATA_FIELDS = (
    "q_d",
    "dq",
    "dq_d",
    "ddq_d",
    "tau_J",
    "tau_J_d",
    "tau_ext_hat_filtered",
    "robot_mode",
    "control_command_success_rate",
    "frame",
    "robot_time_sec",
)


RAW_ROBOT_STATE_FIELDS = (
    "O_T_EE_d",
    "F_T_EE",
    "F_T_NE",
    "NE_T_EE",
    "EE_T_K",
    "elbow",
    "elbow_d",
    "elbow_c",
    "delbow_c",
    "ddelbow_c",
    "tau_J",
    "tau_J_d",
    "dtau_J",
    "q_d",
    "dq",
    "dq_d",
    "ddq_d",
    "joint_contact",
    "cartesian_contact",
    "joint_collision",
    "cartesian_collision",
    "tau_ext_hat_filtered",
    "O_F_ext_hat_K",
    "K_F_ext_hat_K",
    "O_dP_EE_d",
    "O_T_EE_c",
    "O_dP_EE_c",
    "O_ddP_EE_c",
    "theta",
    "dtheta",
    "current_robot_poses_frames",
)


ERROR_FIELDS = (
    "joint_position_limits_violation",
    "cartesian_position_limits_violation",
    "self_collision_avoidance_violation",
    "joint_velocity_violation",
    "cartesian_velocity_violation",
    "force_control_safety_violation",
    "joint_reflex",
    "cartesian_reflex",
    "max_goal_pose_deviation_violation",
    "max_path_pose_deviation_violation",
    "cartesian_velocity_profile_safety_violation",
    "joint_position_motion_generator_start_pose_invalid",
    "joint_motion_generator_position_limits_violation",
    "joint_motion_generator_velocity_limits_violation",
    "joint_motion_generator_velocity_discontinuity",
    "joint_motion_generator_acceleration_discontinuity",
    "cartesian_position_motion_generator_start_pose_invalid",
    "cartesian_motion_generator_elbow_limit_violation",
    "cartesian_motion_generator_velocity_limits_violation",
    "cartesian_motion_generator_velocity_discontinuity",
    "cartesian_motion_generator_acceleration_discontinuity",
    "cartesian_motion_generator_elbow_sign_inconsistent",
    "cartesian_motion_generator_start_elbow_invalid",
    "cartesian_motion_generator_joint_position_limits_violation",
    "cartesian_motion_generator_joint_velocity_limits_violation",
    "cartesian_motion_generator_joint_velocity_discontinuity",
    "cartesian_motion_generator_joint_acceleration_discontinuity",
    "cartesian_position_motion_generator_invalid_frame",
    "force_controller_desired_force_tolerance_violation",
    "controller_torque_discontinuity",
    "start_elbow_sign_inconsistent",
    "communication_constraints_violation",
    "power_limit_violation",
    "joint_p2p_insufficient_torque_for_planning",
    "tau_j_range_violation",
    "instability_detected",
    "joint_move_in_wrong_direction",
)


def apply_hdf5_attrs(dataset, spec: Optional[FieldSpec]) -> None:
    """Apply consistent field metadata to an HDF5 dataset."""

    if spec is None:
        return
    dataset.attrs["units"] = spec.units
    dataset.attrs["description"] = spec.description
    if spec.frame:
        dataset.attrs["frame"] = spec.frame


def field_spec_for(name: str) -> Optional[FieldSpec]:
    """Return a field specification if one is known."""

    for mapping in (CORE_DATASET_SPECS, ROBOT_METADATA_SPECS, CAMERA_METADATA_SPECS):
        if name in mapping:
            return mapping[name]
    return None


def describe_field_units(field_names: Iterable[str]) -> Dict[str, str]:
    """Build a field-to-units mapping for manifests."""

    return {
        field_name: field_spec_for(field_name).units
        for field_name in field_names
        if field_spec_for(field_name) is not None
    }
