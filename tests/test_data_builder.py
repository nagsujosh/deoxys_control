import json
from pathlib import Path

import numpy as np
import pytest

from deoxys.data.builder import HDF5DatasetBuilder
from deoxys.data.config import TaskConfig, load_task_config


def test_builder_writes_expected_core_datasets(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)

    manifest = {
        "status": "success",
        "camera_depth_scales": {"agentview": 0.001, "wrist": 0.001},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    num_samples = 3
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        sample_host_time_sec=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        robot_state_receive_timestamp_sec=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        robot_state_age_sec=np.ones((num_samples, 1), dtype=float),
        gripper_state_receive_timestamp_sec=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        gripper_state_age_sec=np.ones((num_samples, 1), dtype=float),
        agentview_publish_timestamp_sec=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        agentview_acquisition_timestamp_ms=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        agentview_frame_age_sec=np.ones((num_samples, 1), dtype=float),
        agentview_robot_receive_skew_sec=np.ones((num_samples, 1), dtype=float),
        wrist_publish_timestamp_sec=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        wrist_acquisition_timestamp_ms=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
        wrist_frame_age_sec=np.ones((num_samples, 1), dtype=float),
        wrist_robot_receive_skew_sec=np.ones((num_samples, 1), dtype=float),
    )
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_valid.npz",
        data=np.ones((num_samples, 1), dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth_valid.npz",
        data=np.ones((num_samples, 1), dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_valid.npz",
        data=np.array([[1], [0], [1]], dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth_valid.npz",
        data=np.array([[1], [0], [1]], dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array(
            [
                {
                    "frame_id": idx,
                    "acquisition_timestamp_ms": float(idx),
                    "publish_timestamp_sec": float(idx),
                }
                for idx in range(num_samples)
            ],
            dtype=object,
        ),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array(
            [
                {
                    "frame_id": idx,
                    "acquisition_timestamp_ms": float(idx),
                    "publish_timestamp_sec": float(idx),
                }
                for idx in range(num_samples)
            ],
            dtype=object,
        ),
    )
    np.savez_compressed(
        run_dir / "testing_demo_robot_metadata.npz",
        q_d=np.ones((num_samples, 7)),
        dq=np.ones((num_samples, 7)),
        dq_d=np.ones((num_samples, 7)),
        ddq_d=np.ones((num_samples, 7)),
        tau_J=np.ones((num_samples, 7)),
        tau_J_d=np.ones((num_samples, 7)),
        tau_ext_hat_filtered=np.ones((num_samples, 7)),
        robot_mode=np.ones((num_samples, 1)),
        control_command_success_rate=np.ones((num_samples, 1)),
        frame=np.arange(num_samples).reshape(num_samples, 1),
        robot_time_sec=np.arange(num_samples, dtype=float).reshape(num_samples, 1),
    )
    (run_dir / "testing_demo_camera_0_calibration.json").write_text(
        json.dumps(
            {
                "role": "agentview",
                "color_intrinsics": {
                    "native": {"width_px": 640, "height_px": 480, "fx_px": 500.0},
                    "resized": {"width_px": 224, "height_px": 224, "fx_px": 175.0},
                },
                "configured_extrinsics": [
                    {
                        "name": "world_T_agentview",
                        "reference_frame": "world",
                        "target_frame": "agentview_camera",
                        "transform_row_major": [1.0] * 16,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_1_calibration.json").write_text(
        json.dumps(
            {
                "role": "wrist",
                "color_intrinsics": {
                    "native": {"width_px": 640, "height_px": 480, "fx_px": 500.0},
                    "resized": {"width_px": 224, "height_px": 224, "fx_px": 175.0},
                },
            }
        ),
        encoding="utf-8",
    )

    output_path = HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)

    with h5py.File(output_path, "r") as handle:
        demo = handle["data"]["demo_0"]
        assert demo["obs"]["agentview_rgb"].shape == (num_samples, 4, 4, 3)
        assert demo["obs"]["eye_in_hand_depth"].shape == (num_samples, 4, 4)
        assert demo["delta_actions"].shape == (num_samples, 7)
        assert demo["obs"]["ee_state_10d"].shape == (num_samples, 10)
        assert demo["pose_tracking_errors"].shape == (num_samples, 6)
        assert demo["delta_actions"].attrs["description"].startswith("Canonical teleoperation delta action")
        assert demo["meta"]["sync"]["robot_state_age_sec"].shape == (num_samples, 1)
        assert demo["meta"]["camera"]["agentview_valid"].shape == (num_samples, 1)
        assert demo["meta"]["camera"]["agentview_depth_valid"].shape == (num_samples, 1)
        assert demo["meta"]["camera"]["wrist_valid"][1, 0] == 0
        assert demo["meta"]["camera"]["wrist_depth_valid"][1, 0] == 0
        assert demo["meta"]["camera"]["agentview"]["calibration"]["color_intrinsics"]["native"].attrs["fx_px"] == 500.0
        assert handle["data"].attrs["nominal_dataset_rate_hz"] == base_task.control_rates_hz["policy"]


def test_builder_uses_recorded_manifest_task_name_for_root_metadata(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name="renamed_current_task",
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        config_stem=base_task.config_stem,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "task": "recorded_task_name"}),
        encoding="utf-8",
    )

    num_samples = 1
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((num_samples, 7)))

    output_path = HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)

    with h5py.File(output_path, "r") as handle:
        assert handle["data"].attrs["task_name"] == "recorded_task_name"


def test_builder_falls_back_to_info_depth_scale_when_manifest_omits_it(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")

    num_samples = 1
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array(
            [{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0, "depth_scale_m_per_unit": 0.002}],
            dtype=object,
        ),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array(
            [{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0, "depth_scale_m_per_unit": 0.003}],
            dtype=object,
        ),
    )

    output_path = HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)

    with h5py.File(output_path, "r") as handle:
        demo = handle["data"]["demo_0"]
        assert demo["obs"]["agentview_depth"].attrs["depth_scale_m_per_unit"] == pytest.approx(0.002)
        assert demo["obs"]["eye_in_hand_depth"].attrs["depth_scale_m_per_unit"] == pytest.approx(0.003)


def test_builder_uses_run_manifest_camera_layout(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "controller_type": "OSC_YAW",
                "control_rates_hz": {"policy": 11, "state_publisher": 55, "trajectory_interpolation": 444},
                "camera_capture_rates_hz": {"agentview": 17},
                "camera_roles": ["agentview"],
                "camera_ids": {"agentview": 5},
                "depth_enabled": {"agentview": False},
            }
        ),
        encoding="utf-8",
    )

    num_samples = 2
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        robot_state_age_sec=np.ones((num_samples, 1)),
    )
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((num_samples, 7)))
    np.savez_compressed(
        run_dir / "testing_demo_camera_5_color.npz",
        data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_5_valid.npz",
        data=np.ones((num_samples, 1), dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_5_info.npz",
        data=np.array(
            [
                {"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)}
                for idx in range(num_samples)
            ],
            dtype=object,
        ),
    )

    output_path = HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)

    with h5py.File(output_path, "r") as handle:
        data_group = handle["data"]
        demo = handle["data"]["demo_0"]
        assert "agentview_rgb" in demo["obs"]
        assert "eye_in_hand_rgb" not in demo["obs"]
        assert data_group.attrs["controller_type"] == "OSC_YAW"
        assert json.loads(data_group.attrs["control_rates_hz_json"])["policy"] == 11
        assert json.loads(data_group.attrs["camera_capture_rates_hz_json"])["agentview"] == 17
        assert data_group.attrs["nominal_dataset_rate_hz"] == 11


def test_builder_rejects_length_mismatches(tmp_path: Path):
    pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "camera_depth_scales": {"agentview": 0.001, "wrist": 0.001}}),
        encoding="utf-8",
    )

    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((1, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((2, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((2, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (2, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((2, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((2, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((2, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((2, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((2, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((2, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((2, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((2, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array(
            [{"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)} for idx in range(2)],
            dtype=object,
        ),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array(
            [{"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)} for idx in range(2)],
            dtype=object,
        ),
    )

    with pytest.raises(ValueError, match="stream lengths"):
        HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)


def test_builder_ignores_non_numeric_run_directories(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (date_root / "run_backup").mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "camera_depth_scales": {"agentview": 0.001, "wrist": 0.001}}),
        encoding="utf-8",
    )

    num_samples = 1
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        robot_state_age_sec=np.ones((num_samples, 1)),
        gripper_state_age_sec=np.ones((num_samples, 1)),
        agentview_frame_age_sec=np.ones((num_samples, 1)),
        agentview_robot_receive_skew_sec=np.ones((num_samples, 1)),
        wrist_frame_age_sec=np.ones((num_samples, 1)),
        wrist_robot_receive_skew_sec=np.ones((num_samples, 1)),
    )
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((num_samples, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 0, "acquisition_timestamp_ms": 0.0, "publish_timestamp_sec": 0.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 0, "acquisition_timestamp_ms": 0.0, "publish_timestamp_sec": 0.0}], dtype=object),
    )

    output_path = HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)

    with h5py.File(output_path, "r") as handle:
        assert list(handle["data"].keys()) == ["demo_0"]


def test_builder_requires_depth_artifacts_for_depth_enabled_camera(tmp_path: Path):
    pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "camera_depth_scales": {"agentview": 0.001, "wrist": 0.001}}),
        encoding="utf-8",
    )

    num_samples = 1
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 0, "acquisition_timestamp_ms": 0.0, "publish_timestamp_sec": 0.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 0, "acquisition_timestamp_ms": 0.0, "publish_timestamp_sec": 0.0}], dtype=object),
    )

    with pytest.raises(FileNotFoundError, match="missing depth data|missing depth-validity data"):
        HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)


def test_builder_succeeds_without_depth_artifacts_when_depth_disabled(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    depthless_cameras = [
        camera.__class__(
            camera_id=camera.camera_id,
            role=camera.role,
            serial_number=camera.serial_number,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            resize=camera.resize,
            require_rgb=camera.require_rgb,
            enable_depth=False,
            require_depth=False,
            color_encoding=camera.color_encoding,
            depth_encoding=camera.depth_encoding,
            extrinsics=camera.extrinsics,
        )
        for camera in base_task.cameras
    ]
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=depthless_cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")

    num_samples = 1
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        robot_state_age_sec=np.ones((num_samples, 1)),
        gripper_state_age_sec=np.ones((num_samples, 1)),
        agentview_frame_age_sec=np.ones((num_samples, 1)),
        agentview_robot_receive_skew_sec=np.ones((num_samples, 1)),
        wrist_frame_age_sec=np.ones((num_samples, 1)),
        wrist_robot_receive_skew_sec=np.ones((num_samples, 1)),
    )
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((num_samples, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 0, "acquisition_timestamp_ms": 0.0, "publish_timestamp_sec": 0.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 0, "acquisition_timestamp_ms": 0.0, "publish_timestamp_sec": 0.0}], dtype=object),
    )

    output_path = HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)

    with h5py.File(output_path, "r") as handle:
        demo = handle["data"]["demo_0"]
        assert "agentview_depth" not in demo["obs"]
        assert "eye_in_hand_depth" not in demo["obs"]
        assert "agentview_depth_valid" not in demo["meta"]["camera"]
        assert "wrist_depth_valid" not in demo["meta"]["camera"]


def test_builder_rejects_internal_metadata_archive_mismatches(tmp_path: Path):
    pytest.importorskip("h5py", reason="h5py not installed")

    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    date_root = tmp_path / task.name / "2026-03-14"
    run_dir = date_root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "camera_depth_scales": {"agentview": 0.001, "wrist": 0.001}}),
        encoding="utf-8",
    )

    num_samples = 2
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((num_samples, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (num_samples, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((num_samples, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((num_samples, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((num_samples, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        robot_state_age_sec=np.ones((num_samples, 1)),
        gripper_state_age_sec=np.ones((1, 1)),
    )
    np.savez_compressed(
        run_dir / "testing_demo_robot_metadata.npz",
        q_d=np.ones((num_samples, 7)),
        dq=np.ones((1, 7)),
    )
    for camera_id in (0, 1):
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_color.npz",
            data=np.zeros((num_samples, 4, 4, 3), dtype=np.uint8),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_depth.npz",
            data=np.zeros((num_samples, 4, 4), dtype=np.uint16),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_valid.npz",
            data=np.ones((num_samples, 1), dtype=np.uint8),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_depth_valid.npz",
            data=np.ones((num_samples, 1), dtype=np.uint8),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_info.npz",
            data=np.array(
                [
                    {"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)}
                    for idx in range(num_samples)
                ],
                dtype=object,
            ),
        )

    with pytest.raises(ValueError, match="testing_demo_sync_metadata.npz|testing_demo_robot_metadata.npz"):
        HDF5DatasetBuilder(task).build(date_str="2026-03-14", overwrite=True)
