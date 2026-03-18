"""HDF5 dataset builder for raw teleoperation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import TaskConfig
from .logging_utils import get_data_logger
from .metadata import CORE_DATASET_SPECS, apply_hdf5_attrs, field_spec_for
from .paths import default_date_str, get_task_date_root, get_task_root, latest_date_root, list_date_roots
from .run_tools import list_run_dirs, run_camera_specs

logger = get_data_logger("builder")


def _resolve_delta_action_path(run_dir: Path, manifest: Dict[str, object]) -> Path:
    """Resolve the canonical delta-action file, with guarded legacy fallback."""

    canonical = run_dir / "testing_demo_delta_action.npz"
    if canonical.exists():
        return canonical

    legacy = run_dir / "testing_demo_action.npz"
    if legacy.exists() and manifest.get("action_semantics") == "delta_action":
        logger.warning(
            "Using legacy delta-action file %s because manifest still advertises delta_action semantics",
            legacy,
        )
        return legacy
    raise FileNotFoundError(f"No delta-action file found under {run_dir}")


def _write_json_like_hdf5(group, payload):
    """Recursively store JSON-like calibration metadata in HDF5."""

    for key, value in payload.items():
        if isinstance(value, dict):
            child = group.require_group(key)
            _write_json_like_hdf5(child, value)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                child = group.require_group(key)
                for index, item in enumerate(value):
                    item_group = child.require_group(str(index))
                    _write_json_like_hdf5(item_group, item)
            else:
                group.create_dataset(key, data=np.array(value))
        elif isinstance(value, str):
            group.attrs[key] = value
        elif value is None:
            group.attrs[key] = "null"
        else:
            group.attrs[key] = value


def _validate_num_samples(expected: int, arrays: Dict[str, np.ndarray], run_dir: Path) -> None:
    """Fail fast when a successful run contains length-mismatched streams."""

    mismatched = {
        name: int(array.shape[0])
        for name, array in arrays.items()
        if int(array.shape[0]) != expected
    }
    if mismatched:
        details = ", ".join(f"{name}={length}" for name, length in sorted(mismatched.items()))
        raise ValueError(
            f"Run {run_dir} has stream lengths that do not match num_samples={expected}: {details}"
        )


def _depth_scale_for_camera(
    manifest: Dict[str, object], camera_role: str, calibration: Optional[Dict[str, object]], info: Optional[np.ndarray]
) -> float:
    """Resolve a depth scale from manifest first, then calibration, then frame metadata."""

    depth_scales = manifest.get("camera_depth_scales", {})
    if isinstance(depth_scales, dict) and camera_role in depth_scales:
        manifest_value = float(depth_scales.get(camera_role, 0.0))
        if manifest_value > 0.0:
            return manifest_value
    if isinstance(calibration, dict):
        calibration_value = float(calibration.get("depth_scale_m_per_unit", 0.0))
        if calibration_value > 0.0:
            return calibration_value
    if info is not None and len(info) > 0:
        for entry in info:
            value = float(entry.get("depth_scale_m_per_unit", 0.0))
            if np.isfinite(value) and value > 0.0:
                return value
    return 0.0


def _manifest_root_metadata(manifest: Dict[str, object], task_cfg: TaskConfig) -> Dict[str, object]:
    """Resolve root-level dataset metadata from one recorded run manifest."""

    control_rates = manifest.get("control_rates_hz", task_cfg.control_rates_hz)
    if not isinstance(control_rates, dict):
        control_rates = task_cfg.control_rates_hz
    camera_capture_rates = manifest.get(
        "camera_capture_rates_hz", task_cfg.camera_capture_rates_hz
    )
    if not isinstance(camera_capture_rates, dict):
        camera_capture_rates = task_cfg.camera_capture_rates_hz
    camera_roles = manifest.get("camera_roles", task_cfg.camera_roles)
    if not isinstance(camera_roles, list):
        camera_roles = task_cfg.camera_roles
    return {
        "task_name": str(manifest.get("task", task_cfg.name)),
        "controller_type": str(manifest.get("controller_type", task_cfg.controller_type)),
        "control_rates_hz": control_rates,
        "camera_capture_rates_hz": camera_capture_rates,
        "nominal_dataset_rate_hz": int(control_rates.get("policy", 0)),
        "camera_roles": [str(role) for role in camera_roles],
    }


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "The `h5py` package is required to build demo.hdf5 outputs."
        ) from exc
    return h5py


class HDF5DatasetBuilder:
    """Build a canonical HDF5 dataset from raw run directories."""

    def __init__(self, task_cfg: TaskConfig):
        self.task_cfg = task_cfg

    def build(
        self,
        date_str: Optional[str] = None,
        date_strs: Optional[List[str]] = None,
        all_dates: bool = False,
        overwrite: bool = False,
        output_path: str | Path | None = None,
    ) -> Path:
        """Build one `demo.hdf5` file from one or more task/date roots."""

        h5py = _load_h5py()
        task_root = get_task_root(self.task_cfg.output_root, self.task_cfg.name)
        if all_dates and date_strs:
            raise ValueError("Pass either --all-dates or --dates, not both.")

        if all_dates:
            date_roots = list_date_roots(task_root)
            if not date_roots:
                raise FileNotFoundError(f"No date directories found under {task_root}")
        elif date_strs:
            date_roots = [
                get_task_date_root(self.task_cfg.output_root, self.task_cfg.name, item)
                for item in date_strs
            ]
            missing = [path for path in date_roots if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Task/date root not found: {missing[0]}"
                )
        elif date_str is None:
            latest = latest_date_root(task_root)
            if latest is None:
                raise FileNotFoundError(f"No date directories found under {task_root}")
            date_roots = [latest]
        else:
            date_root = get_task_date_root(self.task_cfg.output_root, self.task_cfg.name, date_str)
            if not date_root.exists():
                raise FileNotFoundError(f"Task/date root not found: {date_root}")
            date_roots = [date_root]

        if output_path is None:
            if len(date_roots) > 1:
                output_path = task_root / "demo.hdf5"
            else:
                output_path = date_roots[0] / "demo.hdf5"
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Building HDF5 dataset from %s",
            ", ".join(str(path) for path in date_roots),
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to rebuild it."
            )
        if output_path.exists():
            output_path.unlink()

        run_dirs: List[Path] = []
        for date_root in date_roots:
            run_dirs.extend(list_run_dirs(date_root))
        total = 0
        num_demos = 0
        included_root_manifests: List[Dict[str, object]] = []

        with h5py.File(output_path, "w") as handle:
            data_group = handle.create_group("data")
            for run_dir in run_dirs:
                manifest_path = run_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "success":
                    logger.info("Skipping non-success run %s", run_dir.name)
                    continue
                included_root_manifests.append(_manifest_root_metadata(manifest, self.task_cfg))
                run_cameras = run_camera_specs(self.task_cfg, manifest)

                delta_action = np.load(_resolve_delta_action_path(run_dir, manifest))["data"]
                ee_state_10d = np.load(run_dir / "testing_demo_ee_state_10d.npz")["data"]

                num_samples = int(delta_action.shape[0])
                if num_samples == 0:
                    continue
                _validate_num_samples(
                    num_samples,
                    {
                        "ee_state_10d": ee_state_10d,
                    },
                    run_dir,
                )

                demo_group = data_group.create_group(f"demo_{num_demos}")
                obs_group = demo_group.create_group("obs")

                camera_name_map = {
                    "agentview": "agentview_rgb",
                    "wrist": "eye_in_hand_rgb",
                }
                depth_name_map = {
                    "agentview": "agentview_depth",
                    "wrist": "eye_in_hand_depth",
                }

                for run_camera in run_cameras:
                    color_path = run_dir / f"testing_demo_camera_{run_camera.camera_id}_color.npz"
                    color_data = np.load(color_path)["data"]
                    valid_path = run_dir / f"testing_demo_camera_{run_camera.camera_id}_valid.npz"
                    valid_data = np.load(valid_path)["data"]
                    _validate_num_samples(
                        num_samples,
                        {
                            f"{run_camera.role}_rgb": color_data,
                            f"{run_camera.role}_valid": valid_data,
                        },
                        run_dir,
                    )
                    dataset_name = camera_name_map.get(run_camera.role, run_camera.role)
                    dataset = obs_group.create_dataset(dataset_name, data=color_data)
                    apply_hdf5_attrs(dataset, CORE_DATASET_SPECS[dataset_name])

                    depth_path = run_dir / f"testing_demo_camera_{run_camera.camera_id}_depth.npz"
                    depth_valid_path = run_dir / f"testing_demo_camera_{run_camera.camera_id}_depth_valid.npz"
                    info = None
                    info_path = run_dir / f"testing_demo_camera_{run_camera.camera_id}_info.npz"
                    calibration = None
                    calibration_path = run_dir / f"testing_demo_camera_{run_camera.camera_id}_calibration.json"
                    if info_path.exists():
                        info = np.load(info_path, allow_pickle=True)["data"]
                    if calibration_path.exists():
                        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                    if run_camera.enable_depth and not depth_path.exists():
                        raise FileNotFoundError(
                            f"Depth-enabled camera `{run_camera.role}` is missing depth data in {run_dir}"
                        )
                    if run_camera.enable_depth and not depth_valid_path.exists():
                        raise FileNotFoundError(
                            f"Depth-enabled camera `{run_camera.role}` is missing depth-validity data in {run_dir}"
                        )
                    if depth_path.exists():
                        depth_data = np.load(depth_path)["data"]
                        _validate_num_samples(
                            num_samples,
                            {f"{run_camera.role}_depth": depth_data},
                            run_dir,
                        )
                        dataset_name = depth_name_map.get(run_camera.role, f"{run_camera.role}_depth")
                        dataset = obs_group.create_dataset(dataset_name, data=depth_data)
                        apply_hdf5_attrs(dataset, CORE_DATASET_SPECS[dataset_name])
                        dataset.attrs["depth_scale_m_per_unit"] = _depth_scale_for_camera(
                            manifest, run_camera.role, calibration, info
                        )

                    meta_group = demo_group.require_group("meta").require_group("camera")
                    dataset = meta_group.create_dataset(
                        f"{run_camera.role}_valid", data=valid_data
                    )
                    apply_hdf5_attrs(dataset, field_spec_for("valid"))

                    if depth_valid_path.exists():
                        depth_valid_data = np.load(depth_valid_path)["data"]
                        _validate_num_samples(
                            num_samples,
                            {f"{run_camera.role}_depth_valid": depth_valid_data},
                            run_dir,
                        )
                        meta_group = demo_group.require_group("meta").require_group("camera")
                        dataset = meta_group.create_dataset(
                            f"{run_camera.role}_depth_valid", data=depth_valid_data
                        )
                        apply_hdf5_attrs(dataset, field_spec_for("depth_valid"))

                    if calibration is not None:
                        calibration_group = (
                            demo_group.require_group("meta")
                            .require_group("camera")
                            .require_group(run_camera.role)
                            .require_group("calibration")
                        )
                        _write_json_like_hdf5(calibration_group, calibration)

                dataset = obs_group.create_dataset("ee_state_10d", data=ee_state_10d)
                apply_hdf5_attrs(dataset, CORE_DATASET_SPECS["ee_state_10d"])

                dataset = demo_group.create_dataset("delta_actions", data=delta_action)
                apply_hdf5_attrs(dataset, CORE_DATASET_SPECS["delta_actions"])

                demo_group.attrs["num_samples"] = num_samples
                total += num_samples
                num_demos += 1

            data_group.attrs["num_demos"] = num_demos
            data_group.attrs["total"] = total
            root_metadata = (
                included_root_manifests[0]
                if included_root_manifests
                else _manifest_root_metadata({}, self.task_cfg)
            )
            if any(metadata != root_metadata for metadata in included_root_manifests[1:]):
                logger.warning(
                    "HDF5 root metadata is using the first successful run manifest, but later runs "
                    "in %s have different recorded controller/rate settings",
                    ", ".join(path.name for path in date_roots),
                )
                data_group.attrs["source_manifests_mixed"] = True
            else:
                data_group.attrs["source_manifests_mixed"] = False
            data_group.attrs["root_metadata_source"] = "first_success_run_manifest"
            data_group.attrs["task_name"] = root_metadata["task_name"]
            data_group.attrs["controller_type"] = root_metadata["controller_type"]
            data_group.attrs["control_rates_hz_json"] = json.dumps(
                root_metadata["control_rates_hz"], sort_keys=True
            )
            data_group.attrs["camera_capture_rates_hz_json"] = json.dumps(
                root_metadata["camera_capture_rates_hz"], sort_keys=True
            )
            data_group.attrs["nominal_dataset_rate_hz"] = root_metadata[
                "nominal_dataset_rate_hz"
            ]
            data_group.attrs["action_semantics"] = "controller_delta_action"
            data_group.attrs["action_frame"] = "controller/end-effector command convention"
            data_group.attrs["action_source_device"] = (
                "SpaceMouse mapped into Deoxys controller convention"
            )
            data_group.attrs["camera_roles"] = np.array(root_metadata["camera_roles"], dtype="S")
            data_group.attrs["ee_state_10d_source_field"] = "derived from O_T_EE plus measured gripper width"
        logger.info(
            "Finished HDF5 build at %s with %s demos and %s samples",
            output_path,
            num_demos,
            total,
        )
        return output_path
