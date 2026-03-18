import numpy as np

from deoxys.data.geometry import observed_eef_delta, pose_to_rotation6d, rotation_matrix_to_6d


def test_rotation_matrix_to_6d_uses_first_two_columns():
    rotation = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    result = rotation_matrix_to_6d(rotation)
    np.testing.assert_allclose(result, np.array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0]))


def test_pose_to_rotation6d_identity():
    pose = np.eye(4)
    result = pose_to_rotation6d(pose)
    np.testing.assert_allclose(result, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_observed_eef_delta_translation_only():
    previous_pose = np.eye(4)
    current_pose = np.eye(4)
    current_pose[:3, 3] = np.array([0.1, -0.2, 0.3])
    delta = observed_eef_delta(previous_pose, current_pose)
    np.testing.assert_allclose(delta[:3], np.array([0.1, -0.2, 0.3]))
    np.testing.assert_allclose(delta[3:], np.zeros(3), atol=1e-6)


def test_observed_eef_delta_rotation_about_z():
    previous_pose = np.eye(4)
    current_pose = np.eye(4)
    angle = np.pi / 2.0
    current_pose[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    delta = observed_eef_delta(previous_pose, current_pose)
    np.testing.assert_allclose(delta[:3], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(delta[3:], np.array([0.0, 0.0, angle]), atol=1e-6)
