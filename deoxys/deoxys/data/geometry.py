"""Geometry helpers for dataset feature computation."""

from __future__ import annotations

import numpy as np


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to the standard 6D representation."""

    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    return rotation_matrix[:, :2].reshape(6, order="F")


def pose_to_rotation6d(pose_matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose matrix to a 6D rotation representation."""

    pose_matrix = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    return rotation_matrix_to_6d(pose_matrix[:3, :3])


def pose_to_position_rotation6d(pose_matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose matrix to xyz position plus 6D rotation."""

    pose_matrix = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    return np.concatenate([pose_matrix[:3, 3], pose_to_rotation6d(pose_matrix)])


def _rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a relative rotation matrix to axis-angle.

    The returned vector direction is the rotation axis and its norm is the
    rotation angle in radians. This implementation is pure NumPy so the
    teleoperation collector does not depend on SciPy-backed linear algebra just
    to compute per-sample pose deltas.
    """

    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation_matrix))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    if np.pi - theta < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh(rotation_matrix)
        axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        axis = axis / axis_norm
        return axis * theta

    skew = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis = skew / (2.0 * np.sin(theta))
    return axis * theta


def observed_eef_delta(previous_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    """Compute observed translation and rotation delta between two pose matrices."""

    previous_pose = np.asarray(previous_pose, dtype=np.float64).reshape(4, 4)
    current_pose = np.asarray(current_pose, dtype=np.float64).reshape(4, 4)

    translation_delta = current_pose[:3, 3] - previous_pose[:3, 3]
    relative_rotation = current_pose[:3, :3] @ previous_pose[:3, :3].T
    rotation_delta = _rotation_matrix_to_axis_angle(relative_rotation)
    return np.concatenate([translation_delta, rotation_delta])
