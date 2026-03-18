from pathlib import Path

import numpy as np
from types import SimpleNamespace

from deoxys.data.collector import TeleopCollector
from deoxys.data.cli import build_parser
from deoxys.data.config import (
    TaskConfig,
    create_task_config_from_template,
    load_reset_preset,
    load_task_config,
    normalize_task_name,
)
import pytest

from deoxys.data.paths import allocate_run_dir, get_task_date_root, latest_date_root, next_run_dir
from deoxys.data.run_tools import list_task_dates, list_task_runs, resolve_run_dir


def test_load_task_config_default():
    task = load_task_config("fr3_dual_realsense")
    assert task.name == "fr3_dual_realsense"
    assert task.config_stem == "fr3_dual_realsense"
    assert task.controller_type == "OSC_POSE"
    assert task.default_reset_preset == "home_nominal"
    assert [camera.role for camera in task.cameras] == ["agentview", "wrist"]
    assert task.cameras[0].extrinsics == []
    assert task.cameras[1].extrinsics == []
    assert task.collection.state_zero_fallback is True
    assert task.collection.keep_failed_runs is False
    assert task.collection.action_multipliers == [1.0] * 7
    assert task.collection.optional_camera_roles == []


def test_cli_parser_accepts_inverse_for_teleop_and_collect_only():
    parser = build_parser()

    teleop_args = parser.parse_args(["teleop", "--task", "fr3_dual_realsense", "--inverse"])
    collect_args = parser.parse_args(["collect", "--task", "fr3_dual_realsense", "--inverse"])
    replay_args = parser.parse_args(["replay", "--task", "fr3_dual_realsense"])

    assert teleop_args.inverse is True
    assert collect_args.inverse is True
    assert not hasattr(replay_args, "inverse")


def test_cli_parser_accepts_task_admin_commands():
    parser = build_parser()

    task_create_args = parser.parse_args(
        ["task-create", "--task", "Bell Pepper Pick", "--from-task", "fr3_dual_realsense"]
    )
    dates_args = parser.parse_args(["dates", "--task", "fr3_dual_realsense"])
    runs_args = parser.parse_args(["runs", "--task", "fr3_dual_realsense", "--all-dates"])
    build_args = parser.parse_args(
        ["build", "--task", "fr3_dual_realsense", "--dates", "2026-03-14", "2026-03-15"]
    )

    assert task_create_args.task == "Bell Pepper Pick"
    assert task_create_args.from_task == "fr3_dual_realsense"
    assert dates_args.task == "fr3_dual_realsense"
    assert runs_args.all_dates is True
    assert build_args.dates == ["2026-03-14", "2026-03-15"]


def test_collect_uses_inverseinput2action_when_inverse_enabled(monkeypatch, tmp_path: Path):
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

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            pass

        def control(self, **kwargs):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return SimpleNamespace(is_delta=True)

    class DummySubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("deoxys.data.collector.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.collector.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.collector.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.collector.RedisFrameSubscriber", DummySubscriber)
    monkeypatch.setattr("deoxys.data.collector.ensure_gripper_open", lambda robot: True)
    monkeypatch.setattr("deoxys.data.collector.time.sleep", lambda _value: None)

    calls = {"normal": 0, "inverse": 0}

    monkeypatch.setattr(
        "deoxys.data.collector.input2action",
        lambda **_kwargs: calls.__setitem__("normal", calls["normal"] + 1) or (None, None),
    )
    monkeypatch.setattr(
        "deoxys.data.collector.input2actionInverted",
        lambda **_kwargs: calls.__setitem__("inverse", calls["inverse"] + 1) or (None, None),
    )
    monkeypatch.setattr(TeleopCollector, "_finalize_run", lambda self, **_kwargs: "success")

    result = TeleopCollector(task, inverse=True).collect()

    assert result.status == "success"
    assert calls["normal"] == 0
    assert calls["inverse"] == 1


def test_load_reset_preset_default():
    preset = load_reset_preset("home_nominal")
    assert preset.controller_type == "JOINT_POSITION"
    assert len(preset.joint_positions) == 7
    assert preset.allow_jitter is False


def test_next_run_dir_allocates_monotonic_indices(tmp_path: Path):
    date_root = get_task_date_root(tmp_path, "pick_place", "2026-03-14")
    first = next_run_dir(date_root)
    first.mkdir(parents=True)
    second = next_run_dir(date_root)
    assert first.name == "run1"
    assert second.name == "run2"


def test_allocate_run_dir_creates_directory_and_advances(tmp_path: Path):
    date_root = get_task_date_root(tmp_path, "pick_place", "2026-03-14")

    first = allocate_run_dir(date_root)
    second = allocate_run_dir(date_root)

    assert first.is_dir()
    assert second.is_dir()
    assert first.name == "run1"
    assert second.name == "run2"


def test_latest_date_root_ignores_non_date_directories(tmp_path: Path):
    task_root = tmp_path / "bell_pepper_pick"
    (task_root / "2026-03-14").mkdir(parents=True)
    (task_root / "notes").mkdir()
    (task_root / "2026-03-15").mkdir()

    latest = latest_date_root(task_root)

    assert latest is not None
    assert latest.name == "2026-03-15"


def test_normalize_task_name_and_create_task_from_template(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("deoxys.data.config.DATA_TASKS_ROOT", tmp_path / "tasks")
    monkeypatch.setattr("deoxys.data.config.REPO_ROOT", tmp_path)

    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir(parents=True)
    template_path = tasks_root / "fr3_dual_realsense.yml"
    template_path.write_text(
        "\n".join(
            [
                "name: fr3_dual_realsense",
                "interface_cfg: charmander.yml",
                "controller_type: OSC_POSE",
                "controller_cfg: osc-pose-controller.yml",
                "default_reset_preset: home_nominal",
                "output_root: data",
                "cameras:",
                "  - camera_id: 0",
                "    role: agentview",
            ]
        ),
        encoding="utf-8",
    )

    assert normalize_task_name("Bell Pepper Pick") == "bell_pepper_pick"
    result = create_task_config_from_template("Bell Pepper Pick", template_task="fr3_dual_realsense")

    config_path = Path(result["config_path"])
    assert config_path.exists()
    yaml_text = config_path.read_text(encoding="utf-8")
    assert "name: bell_pepper_pick" in yaml_text
    assert Path(result["task_root"]).exists()


def test_list_task_dates_and_runs(tmp_path: Path):
    base_task = load_task_config("fr3_dual_realsense")
    task = TaskConfig(
        name="bell_pepper_pick",
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

    date_root_a = get_task_date_root(tmp_path, task.name, "2026-03-14")
    run1 = date_root_a / "run1"
    run1.mkdir(parents=True)
    (run1 / "manifest.json").write_text('{"status":"success","num_samples":42}', encoding="utf-8")
    (date_root_a / "demo.hdf5").write_text("", encoding="utf-8")

    date_root_b = get_task_date_root(tmp_path, task.name, "2026-03-15")
    run2 = date_root_b / "run1"
    run2.mkdir(parents=True)
    (run2 / "manifest.json").write_text('{"status":"failed","num_samples":0}', encoding="utf-8")

    dates = list_task_dates(task)
    runs = list_task_runs(task, all_dates=True)

    assert [item["date"] for item in dates] == ["2026-03-14", "2026-03-15"]
    assert dates[0]["has_demo_hdf5"] is True
    assert [item["status"] for item in runs] == ["success", "failed"]


def test_resolve_run_dir_rejects_file_paths(tmp_path: Path):
    base_task = load_task_config("fr3_dual_realsense")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="Expected a run directory path"):
        resolve_run_dir(base_task, run_dir=str(manifest_path))


def test_compute_delta_action_for_non_delta_controller_uses_previous_command():
    action = np.array([2.0, -1.0, 0.5, 0.0, 0.1, -0.2, 1.0])
    previous_command = np.array([1.5, -1.5, 0.0, 0.0, 0.0, -0.1, -1.0])
    delta = TeleopCollector._compute_delta_action(
        action=action,
        previous_command=previous_command,
        controller_is_delta=False,
    )
    assert np.allclose(delta, action - previous_command)


def test_collect_right_button_finishes_and_saves(monkeypatch, tmp_path: Path):
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

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            pass

        def control(self, **kwargs):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return SimpleNamespace(is_delta=True)

    class DummySubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

    actions = iter(
        [
            (np.ones(7, dtype=np.float64), None),
            (None, None),
        ]
    )

    monkeypatch.setattr("deoxys.data.collector.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.collector.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.collector.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.collector.RedisFrameSubscriber", DummySubscriber)
    ensured = []
    monkeypatch.setattr(
        "deoxys.data.collector.ensure_gripper_open",
        lambda robot: ensured.append(robot) or True,
    )
    monkeypatch.setattr("deoxys.data.collector.time.sleep", lambda _value: None)
    monkeypatch.setattr("deoxys.data.collector.input2action", lambda **_kwargs: next(actions))
    emitted_stages = []
    monkeypatch.setattr(
        TeleopCollector,
        "_emit_stage_sound",
        lambda self, stage: emitted_stages.append(stage),
    )

    sample = {
        "delta_action": np.ones(7, dtype=np.float64),
        "controller_command": np.ones(7, dtype=np.float64),
        "ee_state_10d": np.zeros(10, dtype=np.float64),
        "camera_frames": {
            camera.camera_id: {
                "color": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.zeros((2, 2), dtype=np.uint16),
                "calibration": {},
                "valid": True,
                "depth_valid": True,
            }
            for camera in task.cameras
        },
        "valid_state_cache": {
            "ee_pose": np.eye(4, dtype=np.float64),
            "gripper_states": np.zeros(1, dtype=np.float64),
        },
    }

    monkeypatch.setattr(TeleopCollector, "_build_sample", lambda self, **_kwargs: sample)

    finalize_calls = []
    discard_calls = []

    def fake_finalize(self, **kwargs):
        finalize_calls.append(kwargs["run_dir"])
        return "success"

    def fake_discard(self, **kwargs):
        discard_calls.append(kwargs["run_dir"])
        return "discarded"

    monkeypatch.setattr(TeleopCollector, "_finalize_run", fake_finalize)
    monkeypatch.setattr(TeleopCollector, "_discard_run", fake_discard)

    result = TeleopCollector(task).collect()

    assert result.status == "success"
    assert result.num_samples == 1
    assert len(finalize_calls) == 1
    assert discard_calls == []
    assert emitted_stages == ["record_start", "finish_save"]
    assert len(ensured) == 1


def test_collect_keyboard_interrupt_discards_and_clears_run_dir(monkeypatch, tmp_path: Path):
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

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            pass

        def control(self, **kwargs):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return SimpleNamespace(is_delta=True)

    class DummySubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("deoxys.data.collector.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.collector.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.collector.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.collector.RedisFrameSubscriber", DummySubscriber)
    monkeypatch.setattr("deoxys.data.collector.time.sleep", lambda _value: None)
    monkeypatch.setattr(
        "deoxys.data.collector.input2action",
        lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    result = TeleopCollector(task).collect()

    assert result.status == "discarded"
    assert result.run_dir is None
    assert result.failure_reason == "keyboard_interrupt"


def test_collect_returns_failed_status_on_runtime_exception(monkeypatch, tmp_path: Path):
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

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            pass

        def control(self, **kwargs):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return SimpleNamespace(is_delta=True)

    class DummySubscriber:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("deoxys.data.collector.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.collector.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.collector.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.collector.RedisFrameSubscriber", DummySubscriber)
    monkeypatch.setattr("deoxys.data.collector.time.sleep", lambda _value: None)
    monkeypatch.setattr(
        "deoxys.data.collector.input2action",
        lambda **_kwargs: (np.ones(7, dtype=np.float64), None),
    )
    monkeypatch.setattr(
        TeleopCollector,
        "_build_sample",
        lambda self, **_kwargs: (_ for _ in ()).throw(RuntimeError("collect boom")),
    )

    result = TeleopCollector(task).collect()

    assert result.status == "failed"
    assert result.run_dir is None
    assert "collect boom" in result.failure_reason


def test_collect_fails_if_gripper_cannot_be_verified_open(monkeypatch, tmp_path: Path):
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

    class DummyDevice:
        def __init__(self, *args, **kwargs):
            pass

        def start_control(self):
            return None

        def close(self):
            return None

    class DummyRobot:
        def __init__(self, *args, **kwargs):
            pass

        def control(self, **kwargs):
            return None

        def close(self):
            return None

    class DummyYamlConfig:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_easydict(self):
            return SimpleNamespace(is_delta=True)

    monkeypatch.setattr("deoxys.data.collector.SpaceMouse", DummyDevice)
    monkeypatch.setattr("deoxys.data.collector.FrankaInterface", DummyRobot)
    monkeypatch.setattr("deoxys.data.collector.YamlConfig", DummyYamlConfig)
    monkeypatch.setattr("deoxys.data.collector.ensure_gripper_open", lambda robot: False)

    result = TeleopCollector(task).collect()

    assert result.status == "failed"
    assert result.run_dir is None
    assert "gripper_not_open" in result.failure_reason


def test_collect_returns_interrupted_when_reset_is_interrupted(monkeypatch, tmp_path: Path):
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

    def interrupting_reset(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("deoxys.data.reset.run_reset", interrupting_reset)

    result = TeleopCollector(task).collect(preset=load_reset_preset("home_nominal"))

    assert result.status == "interrupted"
    assert result.run_dir is None
    assert result.num_samples == 0


def test_finalize_run_rejects_core_length_mismatches(tmp_path: Path):
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

    collector = TeleopCollector(task)
    run_dir = tmp_path / task.name / "2026-03-15" / "run1"
    run_dir.mkdir(parents=True)
    manifest = collector._initial_manifest(run_dir=run_dir, preset=None)

    data = {
        "delta_action": [np.ones(7, dtype=np.float64)],
        "ee_state_10d": [],
    }
    camera_color = {
        camera.camera_id: [np.zeros((2, 2, 3), dtype=np.uint8)] for camera in task.cameras
    }
    camera_depth = {
        camera.camera_id: [np.zeros((2, 2), dtype=np.uint16)]
        for camera in task.cameras
        if camera.enable_depth
    }
    camera_depth_valid = {
        camera.camera_id: [np.ones(1, dtype=np.uint8)]
        for camera in task.cameras
        if camera.enable_depth
    }
    camera_calibration = {
        camera.camera_id: {"depth_scale_m_per_unit": 0.001} for camera in task.cameras
    }
    camera_valid = {
        camera.camera_id: [np.ones(1, dtype=np.uint8)] for camera in task.cameras
    }

    status = collector._finalize_run(
        run_dir=run_dir,
        manifest=manifest,
        data=data,
        camera_color=camera_color,
        camera_depth=camera_depth,
        camera_depth_valid=camera_depth_valid,
        camera_calibration=camera_calibration,
        camera_valid=camera_valid,
        skip_counters={},
    )

    assert status == "failed"
