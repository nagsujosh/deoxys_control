import json
from pathlib import Path

import numpy as np
import pytest

from deoxys.data.config import TaskConfig, load_task_config
from deoxys.data.validate import validate_run
from deoxys.data.viewer import _stack_panels_vertically


def test_validate_run_reports_lengths_and_camera_stats(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 3, "skip_counters": {}}),
        encoding="utf-8",
    )

    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((3, 6)))
    np.savez_compressed(
        run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((3, 7))
    )
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((3, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((3, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (3, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((3, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((3, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        robot_state_age_sec=np.array([[0.01], [0.02], [0.03]], dtype=np.float64),
        gripper_state_age_sec=np.array([[0.01], [0.02], [0.03]], dtype=np.float64),
        agentview_frame_age_sec=np.array([[0.04], [0.05], [0.06]], dtype=np.float64),
        agentview_robot_receive_skew_sec=np.array([[0.01], [0.01], [0.02]], dtype=np.float64),
        wrist_frame_age_sec=np.array([[0.03], [0.04], [0.05]], dtype=np.float64),
        wrist_robot_receive_skew_sec=np.array([[0.0], [0.01], [0.01]], dtype=np.float64),
    )
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((3, 7)))

    np.savez_compressed(
        run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((3, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((3, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((3, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((3, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_valid.npz", data=np.array([[1], [1], [1]], dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.array([[1], [1], [0]], dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_valid.npz", data=np.array([[1], [0], [1]], dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.array([[1], [0], [1]], dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array(
            [
                {"frame_id": 10, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0},
                {"frame_id": 11, "acquisition_timestamp_ms": 2.0, "publish_timestamp_sec": 2.0},
                {"frame_id": 13, "acquisition_timestamp_ms": 3.0, "publish_timestamp_sec": 3.0},
            ],
            dtype=object,
        ),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array(
            [
                {"frame_id": 20, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0},
                {"frame_id": -1, "acquisition_timestamp_ms": float("nan"), "publish_timestamp_sec": float("nan")},
                {"frame_id": 21, "acquisition_timestamp_ms": 3.0, "publish_timestamp_sec": 3.0},
            ],
            dtype=object,
        ),
    )
    (run_dir / "testing_demo_camera_0_calibration.json").write_text(
        json.dumps({"role": "agentview", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_1_calibration.json").write_text(
        json.dumps({"role": "wrist", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )

    report = validate_run(task, date_str="2026-03-15", run_name="run1")

    assert report["ok"] is True
    assert report["stream_lengths"]["delta_actions"] == 3
    assert report["stream_lengths"]["ee_state_10d"] == 3
    assert report["configured_rates_hz"]["control"]["policy"] == base_task.control_rates_hz["policy"]
    assert report["camera_stats"]["agentview"]["frame_jump_count"] == 1
    assert report["camera_stats"]["wrist"]["missing_or_placeholder_frames"] == 1
    assert report["camera_stats"]["agentview"]["missing_or_placeholder_depth_frames"] == 1
    assert report["timing_stats_sec"]["agentview_frame_age_sec"]["mean"] == pytest.approx(0.05)


def test_validate_uses_run_manifest_camera_layout(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "num_samples": 2,
                "skip_counters": {},
                "control_rates_hz": {"policy": 13, "state_publisher": 99, "trajectory_interpolation": 777},
                "camera_capture_rates_hz": {"agentview": 21},
                "camera_roles": ["agentview"],
                "camera_ids": {"agentview": 5},
                "depth_enabled": {"agentview": False},
            }
        ),
        encoding="utf-8",
    )

    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((2, 6)))
    np.savez_compressed(
        run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((2, 7))
    )
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((2, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((2, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (2, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((2, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((2, 1)))
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((2, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((2, 7)))
    np.savez_compressed(
        run_dir / "testing_demo_camera_5_color.npz", data=np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_5_valid.npz", data=np.ones((2, 1), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_5_info.npz",
        data=np.array(
            [
                {"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0},
                {"frame_id": 2, "acquisition_timestamp_ms": 2.0, "publish_timestamp_sec": 2.0},
            ],
            dtype=object,
        ),
    )
    (run_dir / "testing_demo_camera_5_calibration.json").write_text(
        json.dumps({"role": "agentview"}),
        encoding="utf-8",
    )

    report = validate_run(task, date_str="2026-03-15", run_name="run1")

    assert report["ok"] is True
    assert "agentview_rgb" in report["stream_lengths"]
    assert "wrist_rgb" not in report["stream_lengths"]
    assert report["configured_rates_hz"]["control"]["policy"] == 13
    assert report["configured_rates_hz"]["camera_capture"]["agentview"] == 21


def test_validate_rejects_ambiguous_legacy_action_file(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 1, "skip_counters": {}}),
        encoding="utf-8",
    )

    # Legacy file exists, but the manifest does not advertise delta-action semantics.
    np.savez_compressed(run_dir / "testing_demo_action.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((1, 6)))
    np.savez_compressed(
        run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((1, 7))
    )
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((1, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (1, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((1, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((1, 7)))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((1, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((1, 4, 4, 3), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((1, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((1, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((1, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((1, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((1, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((1, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    (run_dir / "testing_demo_camera_0_calibration.json").write_text(
        json.dumps({"role": "agentview", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_1_calibration.json").write_text(
        json.dumps({"role": "wrist", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_0_calibration.json").write_text(
        json.dumps({"role": "agentview", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_1_calibration.json").write_text(
        json.dumps({"role": "wrist", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )

    report = validate_run(task, date_str="2026-03-15", run_name="run1")

    assert report["ok"] is False
    assert "testing_demo_delta_action.npz" in " ".join(report["issues"])


def test_validate_reports_corrupt_npz_instead_of_crashing(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 1, "skip_counters": {}}),
        encoding="utf-8",
    )

    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((1, 6)))
    np.savez_compressed(
        run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((1, 7))
    )
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((1, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (1, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((1, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((1, 7)))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((1, 1), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((1, 1), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((1, 1), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((1, 1), dtype=np.uint8)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((1, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((1, 4, 4), dtype=np.uint16)
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )

    # Simulate an interrupted save that left a partial/corrupt color archive behind.
    (run_dir / "testing_demo_camera_0_color.npz").write_bytes(b"not-a-valid-npz")
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((1, 4, 4, 3), dtype=np.uint8)
    )

    report = validate_run(task, date_str="2026-03-15", run_name="run1")

    assert report["ok"] is False
    assert any("testing_demo_camera_0_color.npz" in issue for issue in report["issues"])


def test_stack_panels_vertically_pads_mixed_widths():
    narrow = np.zeros((4, 6, 3), dtype=np.uint8)
    wide = np.ones((4, 10, 3), dtype=np.uint8)

    canvas = _stack_panels_vertically([narrow, wide])

    assert canvas.shape == (8, 10, 3)
    assert np.all(canvas[:4, 6:, :] == 0)


def test_validate_playback_skips_placeholder_depth(monkeypatch, tmp_path: Path):
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")

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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 1, "skip_counters": {}}),
        encoding="utf-8",
    )

    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((1, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((1, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (1, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((1, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((1, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((1, 7)))
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((1, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((1, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((1, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((1, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.array([[0]], dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.array([[1]], dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array([{"frame_id": 1, "acquisition_timestamp_ms": 1.0, "publish_timestamp_sec": 1.0}], dtype=object),
    )
    (run_dir / "testing_demo_camera_0_calibration.json").write_text(
        json.dumps({"role": "agentview", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_1_calibration.json").write_text(
        json.dumps({"role": "wrist", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )

    captured = {"shape": None}

    def fake_imshow(_name, image):
        captured["shape"] = image.shape

    monkeypatch.setattr("deoxys.data.validate._load_cv2", lambda: cv2)
    monkeypatch.setattr("cv2.imshow", fake_imshow)
    monkeypatch.setattr("cv2.waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)

    report = validate_run(task, date_str="2026-03-15", run_name="run1", play=True, include_depth=True)

    assert report["ok"] is True
    assert captured["shape"] == (8, 8, 3)


def test_validate_reports_misaligned_metadata_streams(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 3, "skip_counters": {}}),
        encoding="utf-8",
    )

    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((3, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((3, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((3, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (3, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((3, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((3, 1)))
    np.savez_compressed(run_dir / "testing_demo_sync_metadata.npz", robot_state_age_sec=np.ones((2, 1)))
    np.savez_compressed(run_dir / "testing_demo_robot_metadata.npz", q_d=np.ones((3, 7)))
    np.savez_compressed(run_dir / "testing_demo_camera_0_color.npz", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_color.npz", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth.npz", data=np.zeros((3, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth.npz", data=np.zeros((3, 4, 4), dtype=np.uint16))
    np.savez_compressed(run_dir / "testing_demo_camera_0_valid.npz", data=np.ones((2, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_valid.npz", data=np.ones((3, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_0_depth_valid.npz", data=np.ones((3, 1), dtype=np.uint8))
    np.savez_compressed(run_dir / "testing_demo_camera_1_depth_valid.npz", data=np.ones((3, 1), dtype=np.uint8))
    np.savez_compressed(
        run_dir / "testing_demo_camera_0_info.npz",
        data=np.array(
            [{"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)} for idx in range(3)],
            dtype=object,
        ),
    )
    np.savez_compressed(
        run_dir / "testing_demo_camera_1_info.npz",
        data=np.array(
            [{"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)} for idx in range(3)],
            dtype=object,
        ),
    )
    (run_dir / "testing_demo_camera_0_calibration.json").write_text(
        json.dumps({"role": "agentview", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )
    (run_dir / "testing_demo_camera_1_calibration.json").write_text(
        json.dumps({"role": "wrist", "depth_scale_m_per_unit": 0.001}),
        encoding="utf-8",
    )

    report = validate_run(task, date_str="2026-03-15", run_name="run1")

    assert report["ok"] is False
    assert "stream lengths are mismatched" in report["issues"]
    assert report["stream_lengths"]["sync_metadata"] == 2
    assert report["stream_lengths"]["agentview_valid"] == 2


def test_validate_refuses_to_play_invalid_run(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 1, "skip_counters": {}}),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((1, 7)))

    with pytest.raises(RuntimeError, match="Refusing to replay invalid run"):
        validate_run(task, date_str="2026-03-15", run_name="run1", play=True)


def test_validate_reports_internal_metadata_archive_mismatches(tmp_path: Path):
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

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "success", "num_samples": 2, "skip_counters": {}}),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_eef.npz", data=np.ones((2, 6)))
    np.savez_compressed(run_dir / "testing_demo_observed_delta_joint_states.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_pose_tracking_error.npz", data=np.ones((2, 6)))
    np.savez_compressed(run_dir / "testing_demo_joint_tracking_error.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_tracking_error.npz", data=np.ones((2, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_states.npz", data=np.tile(np.eye(4), (2, 1, 1)))
    np.savez_compressed(run_dir / "testing_demo_ee_state_10d.npz", data=np.ones((2, 10)))
    np.savez_compressed(run_dir / "testing_demo_joint_states.npz", data=np.ones((2, 7)))
    np.savez_compressed(run_dir / "testing_demo_gripper_states.npz", data=np.ones((2, 1)))
    np.savez_compressed(
        run_dir / "testing_demo_sync_metadata.npz",
        robot_state_age_sec=np.ones((2, 1)),
        gripper_state_age_sec=np.ones((1, 1)),
    )
    np.savez_compressed(
        run_dir / "testing_demo_robot_metadata.npz",
        q_d=np.ones((2, 7)),
        dq=np.ones((1, 7)),
    )
    for camera_id in (0, 1):
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_color.npz",
            data=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_depth.npz",
            data=np.zeros((2, 4, 4), dtype=np.uint16),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_valid.npz",
            data=np.ones((2, 1), dtype=np.uint8),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_depth_valid.npz",
            data=np.ones((2, 1), dtype=np.uint8),
        )
        np.savez_compressed(
            run_dir / f"testing_demo_camera_{camera_id}_info.npz",
            data=np.array(
                [
                    {"frame_id": idx, "acquisition_timestamp_ms": float(idx), "publish_timestamp_sec": float(idx)}
                    for idx in range(2)
                ],
                dtype=object,
            ),
        )

    report = validate_run(task, date_str="2026-03-15", run_name="run1")

    assert report["ok"] is False
    assert any("testing_demo_sync_metadata.npz has internally mismatched" in issue for issue in report["issues"])
    assert any("testing_demo_robot_metadata.npz has internally mismatched" in issue for issue in report["issues"])
