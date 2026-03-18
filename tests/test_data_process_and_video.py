import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from deoxys.data.control_utils import termination_action_for_controller
from deoxys.data.config import TaskConfig, load_reset_preset, load_task_config
from deoxys.data.camera_node import RealSenseCameraNode
from deoxys.data.process_manager import ProcessManager
from deoxys.data.replay import _policy_step_interval_sec, replay_run
from deoxys.data.redis_io import RedisFramePublisher, RedisFrameSubscriber
from deoxys.data.reset import run_reset
from deoxys.data.teleop import run_teleop
from deoxys.data.video_export import _video_canvas, export_hdf5_demo_videos
from deoxys.data.viewer import _viewer_window_closed
from deoxys.franka_interface import FrankaInterface


def test_health_report_marks_process_liveness_unknown_when_unmanaged(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    class DummyFrame:
        def __init__(self):
            self.info = {"publish_timestamp_sec": 0.0, "frame_id": 1}

    class DummySubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_frame(self):
            return DummyFrame()

    monkeypatch.setattr("deoxys.data.process_manager.RedisFrameSubscriber", DummySubscriber)
    monkeypatch.setattr(ProcessManager, "_redis_ping", lambda self: True)
    monkeypatch.setattr("time.time", lambda: 0.0)

    report = ProcessManager(task).health_report()

    assert report["agentview"]["managed_process_alive"] is None
    assert report["wrist"]["managed_process_alive"] is None


def test_viewer_window_closed_treats_backend_errors_as_closed():
    class DummyCv2:
        WND_PROP_VISIBLE = 4
        WND_PROP_AUTOSIZE = 1

        @staticmethod
        def getWindowProperty(_name, _prop):
            raise RuntimeError("window missing")

    assert _viewer_window_closed(DummyCv2) is True


def test_franka_interface_close_joins_threads_before_terminating_context():
    events = []

    class DummyThread:
        def __init__(self, name):
            self.name = name

        def join(self, timeout):
            events.append((f"join_{self.name}", timeout))

    class DummySocket:
        def __init__(self, name):
            self.name = name

        def close(self, linger=0):
            events.append((f"close_{self.name}", linger))

    class DummyContext:
        def term(self):
            events.append(("term_context", None))

    interface = FrankaInterface.__new__(FrankaInterface)
    interface._closing = False
    interface._state_sub_thread = DummyThread("state")
    interface._gripper_sub_thread = DummyThread("gripper")
    interface._publisher = DummySocket("publisher")
    interface._subscriber = DummySocket("subscriber")
    interface._gripper_publisher = DummySocket("gripper_publisher")
    interface._gripper_subscriber = DummySocket("gripper_subscriber")
    interface._context = DummyContext()

    FrankaInterface.close(interface)

    assert interface._closing is True
    assert events[:2] == [("join_state", 1.0), ("join_gripper", 1.0)]
    assert events[-1] == ("term_context", None)


def test_process_manager_uses_config_stem_for_camera_nodes(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name="bell_pepper_pick",
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        config_stem="fr3_dual_realsense",
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    launched = []

    class DummyProcess:
        def __init__(self, cmd):
            self.cmd = cmd
            self.pid = 1234

        def poll(self):
            return None

    monkeypatch.setattr(
        "deoxys.data.process_manager.subprocess.Popen",
        lambda cmd: launched.append(cmd) or DummyProcess(cmd),
    )
    monkeypatch.setattr(ProcessManager, "_wait_for_process_start", lambda self, _name, grace_sec=0.5: None)

    ProcessManager(task).start_camera_nodes()

    assert launched
    assert "--task" in launched[0]
    assert launched[0][launched[0].index("--task") + 1] == "fr3_dual_realsense"


def test_redis_frame_subscriber_reads_info_and_buffer_with_one_mget(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    camera_cfg = base_task.cameras[0]

    color_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    color_rgb[..., 0] = 255

    class DummyCv2:
        IMREAD_COLOR = 1
        IMREAD_UNCHANGED = -1
        COLOR_BGR2RGB = 4

        @staticmethod
        def imdecode(buffer, flags):
            if flags == DummyCv2.IMREAD_COLOR:
                bgr = np.frombuffer(buffer.tobytes(), dtype=np.uint8).reshape(2, 2, 3)
                return bgr
            return np.frombuffer(buffer.tobytes(), dtype=np.uint16).reshape(2, 2)

        @staticmethod
        def cvtColor(image, _flag):
            return image[..., ::-1]

    class DummyRedis:
        def __init__(self):
            self.get_calls = 0
            self.mget_calls = 0

        def get(self, _key):
            self.get_calls += 1
            raise AssertionError("get() should not be used for frame reads")

        def mget(self, keys):
            self.mget_calls += 1
            assert keys == [
                f"{camera_cfg.redis_namespace}:img_info",
                f"{camera_cfg.redis_namespace}:img_buffer",
            ]
            info_payload = json.dumps({"frame_id": 7}).encode("utf-8")
            bgr = color_rgb[..., ::-1].copy()
            buffer_payload = pickle.dumps({"color": bgr.tobytes()}, protocol=pickle.HIGHEST_PROTOCOL)
            return [info_payload, buffer_payload]

    dummy_redis = DummyRedis()
    monkeypatch.setattr("deoxys.data.redis_io.make_redis_client", lambda _cfg: dummy_redis)
    monkeypatch.setattr("deoxys.data.redis_io._load_cv2", lambda: DummyCv2)

    subscriber = RedisFrameSubscriber(base_task.redis, camera_cfg)
    frame = subscriber.get_frame()

    assert frame is not None
    assert frame.info["frame_id"] == 7
    assert dummy_redis.mget_calls == 1
    assert dummy_redis.get_calls == 0
    np.testing.assert_array_equal(frame.color, color_rgb)


def test_replay_returns_failed_result_when_controller_config_load_fails(monkeypatch, tmp_path: Path):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        config_stem=base_task.config_stem,
        default_reset_preset=base_task.default_reset_preset,
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
        json.dumps({"status": "success", "action_semantics": "delta_action"}),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((2, 7)))

    monkeypatch.setattr(
        "deoxys.data.replay.YamlConfig",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad controller cfg")),
    )

    result = replay_run(task, date_str="2026-03-15", run_name="run1")

    assert result["status"] == "failed"
    assert result["failure_reason"] == "bad controller cfg"
    assert result["replayed_steps"] == 0


def test_redis_frame_publisher_encodes_rgb_input_and_marks_rgb_colorspace(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    camera_cfg = base_task.cameras[0]

    class DummyPipeline:
        def __init__(self):
            self.calls = []

        def set(self, key, value):
            self.calls.append((key, value))
            return self

        def execute(self):
            return None

    class DummyRedis:
        def __init__(self):
            self.pipe = DummyPipeline()

        def pipeline(self):
            return self.pipe

    class DummyCv2:
        COLOR_RGB2BGR = 1

        @staticmethod
        def cvtColor(image, _flag):
            return image[..., ::-1]

        @staticmethod
        def imencode(_ext, image):
            # The publisher should convert RGB input to BGR before encoding.
            assert image[0, 0].tolist() == [30, 20, 10]
            return True, np.frombuffer(b"rgb-jpeg", dtype=np.uint8)

    dummy_redis = DummyRedis()
    monkeypatch.setattr("deoxys.data.redis_io.make_redis_client", lambda _cfg: dummy_redis)
    monkeypatch.setattr("deoxys.data.redis_io._load_cv2", lambda: DummyCv2)
    monkeypatch.setattr("deoxys.data.redis_io.time.time", lambda: 123.0)

    publisher = RedisFramePublisher(base_task.redis, camera_cfg)
    color_rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    publisher.publish(info={"frame_id": 9}, color_image=color_rgb, depth_image=None)

    info_payload = json.loads(dummy_redis.pipe.calls[0][1].decode("utf-8"))
    assert info_payload["color_space"] == "rgb"
    assert info_payload["publish_timestamp_sec"] == 123.0


def test_health_report_uses_unknown_age_for_missing_publish_timestamp(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    class DummyFrame:
        def __init__(self):
            self.info = {"frame_id": 1}

    class DummySubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_frame(self):
            return DummyFrame()

    monkeypatch.setattr("deoxys.data.process_manager.RedisFrameSubscriber", DummySubscriber)
    monkeypatch.setattr(ProcessManager, "_redis_ping", lambda self: True)

    report = ProcessManager(task).health_report()

    assert report["agentview"]["age_sec"] is None
    assert report["agentview"]["fresh"] is False


def test_health_report_treats_optional_camera_as_non_blocking(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    optional_collection = base_task.collection.__class__(
        warmup_sec=base_task.collection.warmup_sec,
        max_samples=base_task.collection.max_samples,
        motion_threshold=base_task.collection.motion_threshold,
        controller_timeout_sec=base_task.collection.controller_timeout_sec,
        state_zero_fallback=base_task.collection.state_zero_fallback,
        state_zero_threshold=base_task.collection.state_zero_threshold,
        keep_failed_runs=base_task.collection.keep_failed_runs,
        action_multipliers=base_task.collection.action_multipliers,
        optional_camera_roles=["wrist"],
    )
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=optional_collection,
        cameras=base_task.cameras,
    )

    class DummyFrame:
        def __init__(self, frame_id):
            self.info = {"publish_timestamp_sec": 10.0, "frame_id": frame_id}

    class DummySubscriber:
        def __init__(self, _redis_cfg, camera_cfg):
            self.role = camera_cfg.role

        def get_frame(self):
            if self.role == "agentview":
                return DummyFrame(1)
            return None

    monkeypatch.setattr("deoxys.data.process_manager.RedisFrameSubscriber", DummySubscriber)
    monkeypatch.setattr(ProcessManager, "_redis_ping", lambda self: True)
    monkeypatch.setattr("time.time", lambda: 10.2)

    report = ProcessManager(task).health_report()
    cameras_healthy = all(
        (
            camera.role in set(task.collection.optional_camera_roles)
            or report.get(camera.role, {}).get("fresh", False)
        )
        for camera in task.cameras
    )

    assert report["agentview"]["fresh"] is True
    assert report["wrist"]["fresh"] is False
    assert cameras_healthy is True


def test_video_export_skips_empty_demo(tmp_path: Path, monkeypatch):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    class DummyWriter:
        def isOpened(self):
            return True

        def write(self, _frame):
            raise AssertionError("writer.write should not be called for empty demos")

        def release(self):
            pass

    class DummyCv2:
        @staticmethod
        def VideoWriter(*_args, **_kwargs):
            return DummyWriter()

        @staticmethod
        def VideoWriter_fourcc(*_args):
            return 0

    monkeypatch.setattr("deoxys.data.video_export._load_cv2", lambda: DummyCv2)

    hdf5_path = tmp_path / "demo.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        demo_group = data_group.create_group("demo_0")
        demo_group.attrs["num_samples"] = 0
        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("agentview_rgb", data=np.zeros((0, 4, 4, 3), dtype=np.uint8))

    exported = export_hdf5_demo_videos(hdf5_path)

    assert exported == []


def test_video_canvas_handles_mixed_depth_panels(monkeypatch):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")

    monkeypatch.setattr("deoxys.data.video_export._load_cv2", lambda: cv2)

    hdf5_path = Path("/tmp/test_video_canvas_mixed_depth.hdf5")
    with h5py.File(hdf5_path, "w") as handle:
        demo_group = handle.create_group("demo")
        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("agentview_rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        obs_group.create_dataset("eye_in_hand_rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        obs_group.create_dataset("agentview_depth", data=np.zeros((1, 4, 4), dtype=np.uint16))

        canvas = _video_canvas(demo_group, 0, include_depth=True)

    assert canvas.shape == (8, 8, 3)


def test_replay_handles_interrupt_before_robot_start(monkeypatch, tmp_path: Path):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
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
                "action_semantics": "delta_action",
                "controller_type": "OSC_POSE",
                "controller_cfg_path": str(task.controller_cfg_path),
                "interface_cfg_path": str(task.interface_cfg_path),
                "control_rates_hz": {"policy": 20},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((2, 7)))

    class InterruptingRobot:
        def __init__(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr("deoxys.data.replay.FrankaInterface", InterruptingRobot)

    result = replay_run(task, date_str="2026-03-15", run_name="run1")

    assert result["status"] == "interrupted"
    assert result["replayed_steps"] == 0
    assert result["elapsed_sec"] == 0.0


def test_replay_uses_manifest_interface_for_reset(monkeypatch, tmp_path: Path):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=str(tmp_path),
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    manifest_interface_cfg = "config/alternate-interface.yml"
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "action_semantics": "delta_action",
                "controller_type": task.controller_type,
                "controller_cfg_path": str(task.controller_cfg_path),
                "interface_cfg": manifest_interface_cfg,
                "interface_cfg_path": str(Path("/home/carl/deoxys_control/deoxys") / manifest_interface_cfg),
                "control_rates_hz": {"policy": 20},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((1, 7)))

    captured = {"interface_cfg": None}

    def fake_run_reset(runtime_task_cfg, _preset):
        captured["interface_cfg"] = runtime_task_cfg.interface_cfg
        return {}

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {"is_delta": True})()

    class DummyRobot:
        def __init__(self, *_args, **_kwargs):
            pass

        def control(self, **_kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr("deoxys.data.reset.run_reset", fake_run_reset)
    monkeypatch.setattr("deoxys.data.replay.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.replay.FrankaInterface", DummyRobot)

    result = replay_run(task, date_str="2026-03-15", run_name="run1", preset=load_reset_preset("home_nominal"))

    assert result["status"] == "success"
    assert captured["interface_cfg"] == manifest_interface_cfg


def test_replay_returns_failed_status_on_runtime_exception(monkeypatch, tmp_path: Path):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
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
                "action_semantics": "delta_action",
                "controller_type": "OSC_POSE",
                "controller_cfg_path": str(task.controller_cfg_path),
                "interface_cfg_path": str(task.interface_cfg_path),
                "control_rates_hz": {"policy": 20},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "testing_demo_delta_action.npz", data=np.ones((2, 7)))

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {"is_delta": True})()

    class FailingRobot:
        def __init__(self, *_args, **_kwargs):
            pass

        def control(self, **_kwargs):
            raise RuntimeError("control boom")

        def close(self):
            return None

    monkeypatch.setattr("deoxys.data.replay.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.replay.FrankaInterface", FailingRobot)

    result = replay_run(task, date_str="2026-03-15", run_name="run1")

    assert result["status"] == "failed"
    assert "control boom" in result["failure_reason"]


def test_video_canvas_marks_placeholder_depth_without_rendering_it(monkeypatch):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")

    monkeypatch.setattr("deoxys.data.video_export._load_cv2", lambda: cv2)

    hdf5_path = Path("/tmp/test_video_canvas_placeholder_depth.hdf5")
    with h5py.File(hdf5_path, "w") as handle:
        demo_group = handle.create_group("demo")
        obs_group = demo_group.create_group("obs")
        meta_camera = demo_group.create_group("meta").create_group("camera")
        obs_group.create_dataset("agentview_rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        obs_group.create_dataset("agentview_depth", data=np.zeros((1, 4, 4), dtype=np.uint16))
        meta_camera.create_dataset("agentview_depth_valid", data=np.array([[0]], dtype=np.uint8))

        canvas = _video_canvas(demo_group, 0, include_depth=True)

    assert canvas.shape == (4, 4, 3)


def test_video_export_rejects_missing_demo_indices(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    hdf5_path = tmp_path / "demo.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        demo_group = data_group.create_group("demo_0")
        demo_group.attrs["num_samples"] = 1
        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("agentview_rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))

    with pytest.raises(KeyError, match="Requested demo indices"):
        export_hdf5_demo_videos(hdf5_path, demo_indices=[1])


def test_video_export_ignores_non_numeric_demo_groups(tmp_path: Path, monkeypatch):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")

    class DummyWriter:
        def __init__(self):
            self.frames = 0

        def isOpened(self):
            return True

        def write(self, _frame):
            self.frames += 1

        def release(self):
            pass

    class DummyCv2:
        COLOR_RGB2BGR = 1

        @staticmethod
        def cvtColor(image, _flag):
            return image

        @staticmethod
        def VideoWriter(*_args, **_kwargs):
            return DummyWriter()

        @staticmethod
        def VideoWriter_fourcc(*_args):
            return 0

    monkeypatch.setattr("deoxys.data.video_export._load_cv2", lambda: DummyCv2)

    hdf5_path = tmp_path / "demo.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data_group = handle.create_group("data")
        backup_group = data_group.create_group("demo_backup")
        backup_group.attrs["num_samples"] = 1
        backup_group.create_group("obs").create_dataset(
            "agentview_rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8)
        )
        demo_group = data_group.create_group("demo_0")
        demo_group.attrs["num_samples"] = 1
        demo_group.create_group("obs").create_dataset(
            "agentview_rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8)
        )

    exported = export_hdf5_demo_videos(hdf5_path)

    assert len(exported) == 1
    assert exported[0].name.endswith("demo_000.mp4")


def test_policy_step_interval_uses_policy_rate():
    assert _policy_step_interval_sec(20) == pytest.approx(0.05)
    assert _policy_step_interval_sec(0) == 0.0


def test_teleop_returns_failed_status_on_runtime_exception(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")

    class DummyDevice:
        def __init__(self, *_args, **_kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {"is_delta": True})()

    class FailingRobot:
        def __init__(self, *_args, **_kwargs):
            pass

        def control(self, **_kwargs):
            raise RuntimeError("teleop boom")

        def close(self):
            return None

    monkeypatch.setattr("deoxys.data.teleop.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.teleop.FrankaInterface", FailingRobot)
    monkeypatch.setattr("deoxys.data.teleop.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr(
        "deoxys.data.teleop.input2action",
        lambda **_kwargs: (np.ones(7, dtype=np.float64), None),
    )

    result = run_teleop(base_task)

    assert result["status"] == "failed"
    assert "teleop boom" in result["failure_reason"]


def test_termination_action_matches_controller_shape():
    assert termination_action_for_controller("OSC_POSE") == [0.0] * 6 + [-1.0]
    assert termination_action_for_controller("JOINT_POSITION") == [0.0] * 7 + [-1.0]


def test_run_teleop_ensures_gripper_open_before_loop(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {"is_delta": True})()

    class DummyRobot:
        def __init__(self, *_args, **_kwargs):
            self.has_gripper = True

        def control(self, **_kwargs):
            return None

        def close(self):
            return None

    ensured = []

    monkeypatch.setattr("deoxys.data.teleop.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.teleop.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.teleop.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.teleop.ensure_gripper_open", lambda robot: ensured.append(robot) or True)
    monkeypatch.setattr("deoxys.data.teleop.input2action", lambda **_kwargs: (None, None))

    result = run_teleop(base_task)

    assert result["status"] == "stopped"
    assert len(ensured) == 1


def test_run_teleop_uses_inverseinput2action_when_inverse_enabled(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {"is_delta": True})()

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            self.has_gripper = True

        def control(self, **_kwargs):
            return None

        def close(self):
            return None

    calls = {"normal": 0, "inverse": 0}

    monkeypatch.setattr("deoxys.data.teleop.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.teleop.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.teleop.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.teleop.ensure_gripper_open", lambda robot: True)
    monkeypatch.setattr(
        "deoxys.data.teleop.input2action",
        lambda **_kwargs: calls.__setitem__("normal", calls["normal"] + 1) or (None, None),
    )
    monkeypatch.setattr(
        "deoxys.data.teleop.input2actionInverted",
        lambda **_kwargs: calls.__setitem__("inverse", calls["inverse"] + 1) or (None, None),
    )

    result = run_teleop(base_task, inverse=True)

    assert result["status"] == "stopped"
    assert calls["normal"] == 0
    assert calls["inverse"] == 1


def test_run_teleop_fails_if_gripper_cannot_be_verified_open(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {"is_delta": True})()

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            self.has_gripper = True

        def control(self, **_kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr("deoxys.data.teleop.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.teleop.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.teleop.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.teleop.ensure_gripper_open", lambda robot: False)

    result = run_teleop(base_task)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "gripper_not_open"


def test_camera_node_republishes_calibration_periodically(monkeypatch):
    published = []
    node = RealSenseCameraNode.__new__(RealSenseCameraNode)
    node.publisher = type(
        "Publisher",
        (),
        {"publish_calibration": lambda self, payload: published.append(payload)},
    )()
    node._calibration_publish_interval_sec = 2.0
    node._last_calibration_publish_time_sec = 0.0

    times = iter([1.0, 1.5, 3.5])
    monkeypatch.setattr("deoxys.data.camera_node.time.time", lambda: next(times))

    payload = {"role": "agentview"}
    node._publish_calibration_if_due(payload, force=True)
    node._publish_calibration_if_due(payload)
    node._publish_calibration_if_due(payload)

    assert published == [payload, payload]


def test_start_camera_nodes_fails_fast_on_early_exit(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    class DummyPopen:
        def __init__(self, *_args, **_kwargs):
            self.returncode = 2
            self.pid = 12345

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("deoxys.data.process_manager.subprocess.Popen", DummyPopen)
    monkeypatch.setattr("time.time", lambda: 0.0)
    monkeypatch.setattr("time.sleep", lambda _value: None)

    manager = ProcessManager(task)
    with pytest.raises(RuntimeError, match="exited early"):
        manager.start_camera_nodes()


def test_start_redis_cleans_up_on_early_exit(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    class DummyPopen:
        def __init__(self, *_args, **_kwargs):
            self.returncode = 2
            self.pid = 999

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("deoxys.data.process_manager.subprocess.Popen", DummyPopen)
    monkeypatch.setattr("time.time", lambda: 0.0)
    monkeypatch.setattr("time.sleep", lambda _value: None)

    manager = ProcessManager(task)
    with pytest.raises(RuntimeError, match="redis.*exited early|Managed process `redis` exited early"):
        manager.start_redis()
    assert "redis" not in manager._processes


def test_start_redis_rechecks_managed_process_after_redis_becomes_reachable(monkeypatch):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name=base_task.name,
        interface_cfg=base_task.interface_cfg,
        controller_type=base_task.controller_type,
        controller_cfg=base_task.controller_cfg,
        default_reset_preset=base_task.default_reset_preset,
        output_root=base_task.output_root,
        spacemouse_vendor_id=base_task.spacemouse_vendor_id,
        spacemouse_product_id=base_task.spacemouse_product_id,
        redis=base_task.redis,
        collection=base_task.collection,
        cameras=base_task.cameras,
    )

    class DummyPopen:
        def __init__(self, *_args, **_kwargs):
            self.returncode = None
            self.pid = 101

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 15

        def wait(self, timeout=None):
            return self.returncode

    wait_calls = {"count": 0}

    def fake_wait_for_process_start(self, process_name, grace_sec=0.5):
        wait_calls["count"] += 1
        if process_name == "redis" and wait_calls["count"] == 2:
            raise RuntimeError("Managed process `redis` exited early with code 1.")

    monkeypatch.setattr("deoxys.data.process_manager.subprocess.Popen", DummyPopen)
    monkeypatch.setattr(ProcessManager, "_wait_for_process_start", fake_wait_for_process_start)
    monkeypatch.setattr(ProcessManager, "_wait_for_redis", lambda self: None)

    manager = ProcessManager(task)
    with pytest.raises(RuntimeError, match="redis.*exited early"):
        manager.start_redis()
    assert "redis" not in manager._processes


def test_run_reset_times_out(monkeypatch):
    task = load_task_config("fr3_dual_realsense")
    reset_preset = load_reset_preset("home_nominal")

    class DummyRobot:
        state_buffer_size = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def control(self, **_kwargs):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return type("Cfg", (), {})()

    monotonic_values = iter([0.0, 31.0])
    monkeypatch.setattr("deoxys.data.reset.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.reset.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.reset.time.monotonic", lambda: next(monotonic_values))

    with pytest.raises(TimeoutError, match="did not converge"):
        run_reset(task, reset_preset)
