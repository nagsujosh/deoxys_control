"""Config loading for the teleoperation data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import yaml

from deoxys import config_root


DATA_TASKS_ROOT = Path(config_root) / "data_tasks"
DATA_RESETS_ROOT = Path(config_root) / "data_resets"
REPO_ROOT = DATA_TASKS_ROOT.parents[2]


@dataclass(frozen=True)
class ResizeConfig:
    """Output image resolution in pixels."""

    width: int
    height: int


@dataclass(frozen=True)
class CameraExtrinsicConfig:
    """One configured camera extrinsic transform.

    The transform is a 4x4 homogeneous matrix stored in row-major order.
    Translation units are meters. Rotation is unitless.
    """

    name: str
    reference_frame: str
    target_frame: str
    transform: List[float] = field(default_factory=list)
    convention: str = "reference_T_target row-major 4x4 homogeneous transform"


@dataclass(frozen=True)
class CameraConfig:
    """Camera configuration for one fixed role."""

    camera_id: int
    role: str
    serial_number: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    resize: ResizeConfig = field(default_factory=lambda: ResizeConfig(width=224, height=224))
    require_rgb: bool = True
    enable_depth: bool = True
    require_depth: bool = False
    color_encoding: str = "jpeg"
    depth_encoding: str = "png"
    extrinsics: List[CameraExtrinsicConfig] = field(default_factory=list)

    @property
    def redis_namespace(self) -> str:
        return f"deoxys:camera:{self.role}"


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection and process management settings."""

    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    start_managed: bool = True
    command: List[str] = field(
        default_factory=lambda: ["redis-server", "--save", "", "--appendonly", "no"]
    )
    freshness_timeout_sec: float = 1.0


@dataclass(frozen=True)
class CollectionConfig:
    """Teleoperation collection behavior."""

    warmup_sec: float = 2.0
    max_samples: int = 0
    motion_threshold: float = 1e-3
    controller_timeout_sec: float = 0.2
    state_zero_fallback: bool = True
    state_zero_threshold: float = 1e-6
    keep_failed_runs: bool = False
    action_multipliers: List[float] = field(default_factory=lambda: [1.0] * 7)
    optional_camera_roles: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskConfig:
    """One fully resolved task definition."""

    name: str
    interface_cfg: str
    controller_type: str
    controller_cfg: str
    config_stem: str = ""
    config_path: str = ""
    default_reset_preset: str = ""
    output_root: str = "data"
    spacemouse_vendor_id: int = 9583
    spacemouse_product_id: int = 50734
    redis: RedisConfig = field(default_factory=RedisConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    cameras: List[CameraConfig] = field(default_factory=list)

    @property
    def interface_cfg_path(self) -> Path:
        return Path(config_root) / self.interface_cfg

    @property
    def controller_cfg_path(self) -> Path:
        return Path(config_root) / self.controller_cfg

    @property
    def camera_roles(self) -> List[str]:
        return [camera.role for camera in self.cameras]

    @property
    def control_rates_hz(self) -> Dict[str, int]:
        """Control-loop rates from the interface config.

        These values come from `config/charmander.yml` and are useful for
        documenting how quickly robot state, policy commands, and trajectory
        interpolation run during teleoperation and replay.
        """

        payload = _load_yaml(self.interface_cfg_path)
        control = payload.get("CONTROL", {})
        return {
            "state_publisher": int(control.get("STATE_PUBLISHER_RATE", 0)),
            "policy": int(control.get("POLICY_RATE", 0)),
            "trajectory_interpolation": int(control.get("TRAJ_RATE", 0)),
        }

    @property
    def camera_capture_rates_hz(self) -> Dict[str, int]:
        """Configured per-camera capture rates in Hz."""

        return {camera.role: int(camera.fps) for camera in self.cameras}

    def to_manifest(self) -> Dict[str, Any]:
        """Convert the task config to a manifest-friendly dictionary."""
        payload = asdict(self)
        payload["interface_cfg_path"] = str(self.interface_cfg_path)
        payload["controller_cfg_path"] = str(self.controller_cfg_path)
        payload["control_rates_hz"] = self.control_rates_hz
        payload["camera_capture_rates_hz"] = self.camera_capture_rates_hz
        return payload


@dataclass(frozen=True)
class ResetPreset:
    """One named reset preset."""

    name: str
    controller_type: str = "JOINT_POSITION"
    controller_cfg: str = "joint-position-controller.yml"
    joint_positions: List[float] = field(default_factory=list)
    tolerance_rad: float = 1e-3
    timeout_sec: float = 30.0
    allow_jitter: bool = False
    jitter_std_rad: float = 0.0
    jitter_clip_rad: float = 0.0

    @property
    def controller_cfg_path(self) -> Path:
        return Path(config_root) / self.controller_cfg

    def to_manifest(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["controller_cfg_path"] = str(self.controller_cfg_path)
        return payload


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _camera_from_dict(payload: Dict[str, Any]) -> CameraConfig:
    resize_payload = payload.get("resize", {})
    extrinsics = [
        CameraExtrinsicConfig(
            name=str(item["name"]),
            reference_frame=str(item["reference_frame"]),
            target_frame=str(item["target_frame"]),
            transform=[float(value) for value in item.get("transform", [])],
            convention=str(
                item.get(
                    "convention",
                    "reference_T_target row-major 4x4 homogeneous transform",
                )
            ),
        )
        for item in payload.get("extrinsics", [])
    ]
    return CameraConfig(
        camera_id=int(payload["camera_id"]),
        role=str(payload["role"]),
        serial_number=str(payload.get("serial_number", "")),
        width=int(payload.get("width", 640)),
        height=int(payload.get("height", 480)),
        fps=int(payload.get("fps", 30)),
        resize=ResizeConfig(
            width=int(resize_payload.get("width", 224)),
            height=int(resize_payload.get("height", 224)),
        ),
        require_rgb=bool(payload.get("require_rgb", True)),
        enable_depth=bool(payload.get("enable_depth", True)),
        require_depth=bool(payload.get("require_depth", False)),
        color_encoding=str(payload.get("color_encoding", "jpeg")),
        depth_encoding=str(payload.get("depth_encoding", "png")),
        extrinsics=extrinsics,
    )


def _task_config_from_payload(
    payload: Dict[str, Any],
    *,
    task_name: str,
    task_path: Path,
) -> TaskConfig:
    """Build one task config from an already-loaded YAML payload."""

    redis_payload = payload.get("redis", {})
    collection_payload = payload.get("collection", {})
    cameras = [_camera_from_dict(item) for item in payload.get("cameras", [])]
    if not cameras:
        raise ValueError(f"Task `{task_name}` must define at least one camera in {task_path}.")

    return TaskConfig(
        name=str(payload.get("name", task_name)),
        config_stem=str(task_name),
        config_path=str(task_path.resolve()),
        interface_cfg=str(payload["interface_cfg"]),
        controller_type=str(payload["controller_type"]),
        controller_cfg=str(payload["controller_cfg"]),
        default_reset_preset=str(payload.get("default_reset_preset", "")),
        output_root=str(payload.get("output_root", "data")),
        spacemouse_vendor_id=int(payload.get("spacemouse_vendor_id", 9583)),
        spacemouse_product_id=int(payload.get("spacemouse_product_id", 50734)),
        redis=RedisConfig(
            host=str(redis_payload.get("host", "127.0.0.1")),
            port=int(redis_payload.get("port", 6379)),
            db=int(redis_payload.get("db", 0)),
            start_managed=bool(redis_payload.get("start_managed", True)),
            command=list(
                redis_payload.get(
                    "command",
                    ["redis-server", "--save", "", "--appendonly", "no"],
                )
            ),
            freshness_timeout_sec=float(redis_payload.get("freshness_timeout_sec", 1.0)),
        ),
        collection=CollectionConfig(
            warmup_sec=float(collection_payload.get("warmup_sec", 2.0)),
            max_samples=int(collection_payload.get("max_samples", 0)),
            motion_threshold=float(collection_payload.get("motion_threshold", 1e-3)),
            controller_timeout_sec=float(
                collection_payload.get("controller_timeout_sec", 0.2)
            ),
            state_zero_fallback=bool(collection_payload.get("state_zero_fallback", True)),
            state_zero_threshold=float(
                collection_payload.get("state_zero_threshold", 1e-6)
            ),
            keep_failed_runs=bool(collection_payload.get("keep_failed_runs", False)),
            action_multipliers=[
                float(value)
                for value in collection_payload.get("action_multipliers", [1.0] * 7)
            ],
            optional_camera_roles=[
                str(value) for value in collection_payload.get("optional_camera_roles", [])
            ],
        ),
        cameras=cameras,
    )


def load_task_config_from_path(config_path: str | Path) -> TaskConfig:
    """Load one task config from an explicit YAML path."""

    task_path = Path(config_path).expanduser().resolve()
    if not task_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_path}")
    payload = _load_yaml(task_path)
    return _task_config_from_payload(payload, task_name=task_path.stem, task_path=task_path)


def load_task_config(task_name: str) -> TaskConfig:
    """Load one versioned task config by stem name."""

    task_path = DATA_TASKS_ROOT / f"{task_name}.yml"
    if not task_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_path}")
    payload = _load_yaml(task_path)
    return _task_config_from_payload(payload, task_name=task_name, task_path=task_path)


def list_task_config_stems() -> List[str]:
    """Return available shared task-profile stems."""

    return sorted(path.stem for path in DATA_TASKS_ROOT.glob("*.yml"))


def normalize_task_name(task_name: str) -> str:
    """Normalize one CLI-provided task name into a filesystem-safe stem."""

    normalized = re.sub(r"[^a-z0-9_-]+", "_", task_name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Task name must contain at least one letter or digit.")
    return normalized


def create_task_config_from_template(
    task_name: str,
    *,
    template_task: str = "fr3_dual_realsense",
    dataset_name: str | None = None,
    output_root: str | None = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Create one new task YAML by copying an existing shared task template."""

    task_stem = normalize_task_name(task_name)
    template_path = DATA_TASKS_ROOT / f"{template_task}.yml"
    if not template_path.exists():
        raise FileNotFoundError(f"Template task config not found: {template_path}")

    payload = _load_yaml(template_path)
    payload["name"] = str(dataset_name or task_stem)
    if output_root is not None:
        payload["output_root"] = str(output_root)

    destination = DATA_TASKS_ROOT / f"{task_stem}.yml"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Task config already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    output_root_value = str(payload.get("output_root", "data"))
    output_root_path = Path(output_root_value).expanduser()
    if not output_root_path.is_absolute():
        output_root_path = (REPO_ROOT / output_root_path).resolve()
    data_root = output_root_path / str(payload["name"])
    data_root.mkdir(parents=True, exist_ok=True)
    return {
        "task": task_stem,
        "config_path": str(destination.resolve()),
        "dataset_name": str(payload["name"]),
        "task_root": str(data_root),
        "template_task": template_task,
    }


def load_reset_preset(preset_name: str) -> ResetPreset:
    """Load one versioned reset preset by stem name."""

    preset_path = DATA_RESETS_ROOT / f"{preset_name}.yml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Reset preset not found: {preset_path}")
    return load_reset_preset_from_path(preset_path)


def load_reset_preset_from_path(preset_path: str | Path) -> ResetPreset:
    """Load one reset preset from an explicit YAML path."""

    preset_path = Path(preset_path).expanduser().resolve()
    if not preset_path.exists():
        raise FileNotFoundError(f"Reset preset not found: {preset_path}")
    payload = _load_yaml(preset_path)
    return ResetPreset(
        name=str(payload.get("name", preset_path.stem)),
        controller_type=str(payload.get("controller_type", "JOINT_POSITION")),
        controller_cfg=str(payload.get("controller_cfg", "joint-position-controller.yml")),
        joint_positions=[float(value) for value in payload.get("joint_positions", [])],
        tolerance_rad=float(payload.get("tolerance_rad", 1e-3)),
        timeout_sec=float(payload.get("timeout_sec", 30.0)),
        allow_jitter=bool(payload.get("allow_jitter", False)),
        jitter_std_rad=float(payload.get("jitter_std_rad", 0.0)),
        jitter_clip_rad=float(payload.get("jitter_clip_rad", 0.0)),
    )
