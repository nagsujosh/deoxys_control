"""Teleoperation data collector with validation and rich metadata."""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.input_utils import input2action, input2actionInverted
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.yaml_config import YamlConfig

from .config import ResetPreset, TaskConfig
from .control_utils import ensure_gripper_open, termination_action_for_controller
from .geometry import pose_to_position_rotation6d
from .logging_utils import get_data_logger
from .metadata import describe_field_units
from .paths import allocate_run_dir, default_date_str, get_task_date_root
from .redis_io import RedisFrameSubscriber

logger = get_data_logger("collector")


class CollectorState(Enum):
    """Explicit collector state machine."""

    INIT = "INIT"
    WARMUP = "WARMUP"
    WAIT_MOTION = "WAIT_MOTION"
    RECORD = "RECORD"
    STOP = "STOP"
    SAVE = "SAVE"


@dataclass
class CollectionResult:
    """Result summary for one raw run."""

    run_dir: Optional[Path]
    status: str
    num_samples: int
    failure_reason: str = ""


class TeleopCollector:
    """Collect validated RGB-D teleoperation runs."""

    def __init__(self, task_cfg: TaskConfig, inverse: bool = False):
        self.task_cfg = task_cfg
        self.inverse = inverse

    def collect(
        self,
        preset: Optional[ResetPreset] = None,
        run_dir: str | Path | None = None,
    ) -> CollectionResult:
        """Collect one run until interrupted or the configured sample limit is reached."""

        run_dir: Optional[Path] = None
        if preset is not None:
            from .reset import run_reset

            logger.info("Running reset preset `%s` before collection", preset.name)
            try:
                run_reset(self.task_cfg, preset)
            except KeyboardInterrupt:
                logger.info("Discarding collection because the reset preset was interrupted")
                return CollectionResult(
                    run_dir=None,
                    status="interrupted",
                    num_samples=0,
                    failure_reason="reset_interrupted",
                )
            except Exception as exc:
                logger.error(
                    "Collection aborted because reset preset `%s` failed: %s", preset.name, exc
                )
                return CollectionResult(
                    run_dir=None,
                    status="failed",
                    num_samples=0,
                    failure_reason=str(exc),
                )

        if run_dir is not None:
            run_dir = Path(run_dir).expanduser().resolve()
            run_dir.mkdir(parents=True, exist_ok=False)
        else:
            date_root = get_task_date_root(
                self.task_cfg.output_root, self.task_cfg.name, default_date_str()
            )
            run_dir = allocate_run_dir(date_root)

        manifest = self._initial_manifest(run_dir=run_dir, preset=preset)
        self._write_manifest(run_dir, manifest)
        logger.info(
            "Starting teleoperation collection in %s with policy_hz=%s state_hz=%s camera_hz=%s",
            run_dir,
            self.task_cfg.control_rates_hz.get("policy", 0),
            self.task_cfg.control_rates_hz.get("state_publisher", 0),
            self.task_cfg.camera_capture_rates_hz,
        )

        device = None
        robot_interface = None
        camera_subscribers = {}
        state = CollectorState.INIT
        count = 0
        skip_counters: Dict[str, int] = {
            "missing_robot_state": 0,
            "missing_gripper_state": 0,
            "missing_rgb": 0,
            "missing_depth": 0,
            "no_motion_yet": 0,
            "invalid_action": 0,
            "spacemouse_finish_signal": 0,
        }

        data: Dict[str, List[np.ndarray]] = {
            "delta_action": [],
            "ee_state_10d": [],
        }
        camera_color: Dict[int, List[np.ndarray]] = {}
        camera_depth: Dict[int, List[np.ndarray]] = {}
        camera_depth_valid: Dict[int, List[np.ndarray]] = {}
        camera_calibration: Dict[int, dict] = {}
        camera_valid: Dict[int, List[np.ndarray]] = {}

        previous_command = None
        previous_valid_state: Optional[Dict[str, np.ndarray]] = None
        interrupted = False
        interrupted_reason = ""
        failed_early = False
        failure_reason = ""
        controller_cfg = None

        try:
            device = SpaceMouse(
                vendor_id=self.task_cfg.spacemouse_vendor_id,
                product_id=self.task_cfg.spacemouse_product_id,
            )
            device.start_control()

            robot_interface = FrankaInterface(
                str(self.task_cfg.interface_cfg_path),
                control_timeout=self.task_cfg.collection.controller_timeout_sec,
                use_visualizer=False,
            )
            controller_cfg = YamlConfig(str(self.task_cfg.controller_cfg_path)).as_easydict()
            logger.info("Ensuring the gripper is fully open before collection starts")
            if not ensure_gripper_open(robot_interface):
                raise RuntimeError(
                    "gripper_not_open: the gripper could not be verified fully open before collection start"
                )

            for camera_cfg in self.task_cfg.cameras:
                camera_subscribers[camera_cfg.camera_id] = RedisFrameSubscriber(
                    self.task_cfg.redis, camera_cfg
                )
                camera_color[camera_cfg.camera_id] = []
                camera_depth[camera_cfg.camera_id] = []
                camera_depth_valid[camera_cfg.camera_id] = []
                camera_calibration[camera_cfg.camera_id] = {}
                camera_valid[camera_cfg.camera_id] = []

            state = CollectorState.WARMUP
            logger.info(
                "Warmup for %.2f seconds before accepting teleoperation samples",
                self.task_cfg.collection.warmup_sec,
            )
            time.sleep(self.task_cfg.collection.warmup_sec)
            state = CollectorState.WAIT_MOTION
            logger.info("Collector is ready and waiting for first non-trivial motion")

            action_fn = input2actionInverted if self.inverse else input2action
            while True:
                action, _ = action_fn(
                    device=device, controller_type=self.task_cfg.controller_type
                )
                if action is None:
                    skip_counters["spacemouse_finish_signal"] += 1
                    self._emit_stage_sound("finish_save")
                    logger.info("Received SpaceMouse finish signal, stopping collection and saving the demo")
                    state = CollectorState.STOP
                    break

                action = np.asarray(action, dtype=np.float64)
                if action.ndim != 1 or action.shape[0] < 7:
                    skip_counters["invalid_action"] += 1
                    logger.warning(
                        "Skipping malformed teleoperation action with shape %s",
                        action.shape,
                    )
                    continue
                action = self._apply_action_transform(action)

                if self.task_cfg.controller_type == "OSC_YAW":
                    action[3:5] = 0.0
                elif self.task_cfg.controller_type == "OSC_POSITION":
                    action[3:6] = 0.0

                sample = self._build_sample(
                    action=action,
                    previous_command=previous_command,
                    previous_valid_state=previous_valid_state,
                    robot_interface=robot_interface,
                    camera_subscribers=camera_subscribers,
                    controller_cfg=controller_cfg,
                    skip_counters=skip_counters,
                )
                if sample is None:
                    continue

                if (
                    state == CollectorState.WAIT_MOTION
                    and np.linalg.norm(action[:-1]) < self.task_cfg.collection.motion_threshold
                ):
                    skip_counters["no_motion_yet"] += 1
                    continue

                if state == CollectorState.WAIT_MOTION:
                    self._emit_stage_sound("record_start")
                    logger.info("Detected first valid motion, switching to RECORD state")
                state = CollectorState.RECORD
                previous_command = sample["controller_command"]
                previous_valid_state = sample["valid_state_cache"]

                data["delta_action"].append(sample["delta_action"])
                data["ee_state_10d"].append(sample["ee_state_10d"])

                for camera_cfg in self.task_cfg.cameras:
                    camera_id = camera_cfg.camera_id
                    camera_color[camera_id].append(sample["camera_frames"][camera_id]["color"])
                    camera_valid[camera_id].append(
                        np.array([sample["camera_frames"][camera_id]["valid"]], dtype=np.uint8)
                    )
                    if not camera_calibration[camera_id]:
                        camera_calibration[camera_id] = sample["camera_frames"][camera_id].get(
                            "calibration", {}
                        )
                    if camera_cfg.enable_depth:
                        camera_depth[camera_id].append(sample["camera_frames"][camera_id]["depth"])
                        camera_depth_valid[camera_id].append(
                            np.array([sample["camera_frames"][camera_id]["depth_valid"]], dtype=np.uint8)
                        )

                count += 1
                if count == 1 or count % 100 == 0:
                    logger.info("Accepted %s samples so far", count)

                # The command is sent after the sample is accepted so the run keeps the
                # imitation-learning contract `(obs_t, delta_action_t)` rather than mixing in
                # post-command observations from `t + 1`.
                robot_interface.control(
                    controller_type=self.task_cfg.controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )

                if self.task_cfg.collection.max_samples > 0 and count >= self.task_cfg.collection.max_samples:
                    state = CollectorState.STOP
                    break
        except KeyboardInterrupt:
            logger.info("Stopping teleoperation collection on keyboard interrupt")
            interrupted = True
            interrupted_reason = "keyboard_interrupt"
            state = CollectorState.STOP
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failure_reason"] = str(exc)
            manifest["skip_counters"] = skip_counters
            failed_early = True
            failure_reason = str(exc)
            if self.task_cfg.collection.keep_failed_runs:
                self._write_manifest(run_dir, manifest)
            else:
                shutil.rmtree(run_dir, ignore_errors=True)
            logger.exception("Teleoperation collection failed: %s", exc)
        finally:
            if robot_interface is not None:
                try:
                    if controller_cfg is not None:
                        robot_interface.control(
                            controller_type=self.task_cfg.controller_type,
                            action=termination_action_for_controller(
                                self.task_cfg.controller_type
                            ),
                            controller_cfg=controller_cfg,
                            termination=True,
                        )
                except Exception as exc:
                    logger.warning("Failed to send collection termination command: %s", exc)
                finally:
                    robot_interface.close()
            if device is not None:
                device.close()

        state = CollectorState.SAVE
        if interrupted:
            status = self._discard_run(
                run_dir=run_dir,
                manifest=manifest,
                skip_counters=skip_counters,
                reason=interrupted_reason,
            )
            logger.info(
                "Discarded interrupted or canceled run output=%s without saving partial data",
                run_dir,
            )
            return CollectionResult(
                run_dir=None,
                status=status,
                num_samples=0,
                failure_reason=interrupted_reason,
            )
        if failed_early:
            self._log_run_summary(status="failed", num_samples=count, skip_counters=skip_counters)
            return CollectionResult(
                run_dir=run_dir if self.task_cfg.collection.keep_failed_runs else None,
                status="failed",
                num_samples=count,
                failure_reason=failure_reason,
            )

        status = self._finalize_run(
            run_dir=run_dir,
            manifest=manifest,
            data=data,
            camera_color=camera_color,
            camera_depth=camera_depth,
            camera_depth_valid=camera_depth_valid,
            camera_calibration=camera_calibration,
            camera_valid=camera_valid,
            skip_counters=skip_counters,
        )
        logger.info(
            "Collection finished with status=%s num_samples=%s output=%s",
            status,
            count,
            run_dir,
        )
        result_run_dir = run_dir
        if status == "failed" and not self.task_cfg.collection.keep_failed_runs:
            shutil.rmtree(run_dir, ignore_errors=True)
            result_run_dir = None
            if not failure_reason:
                failure_reason = "collection_failed"
        self._log_run_summary(status=status, num_samples=count, skip_counters=skip_counters)
        return CollectionResult(
            run_dir=result_run_dir,
            status=status,
            num_samples=count,
            failure_reason=failure_reason,
        )

    def _emit_stage_sound(self, stage: str) -> None:
        """Play lightweight terminal-bell cues for operator-visible stage changes."""

        pattern_by_stage = {
            "record_start": (0.0,),
            "finish_save": (0.0, 0.12),
        }
        pattern = pattern_by_stage.get(stage)
        if pattern is None:
            return

        def _play_pattern() -> None:
            for delay_sec in pattern:
                if delay_sec > 0.0:
                    time.sleep(delay_sec)
                print("\a", end="", flush=True)

        threading.Thread(target=_play_pattern, daemon=True).start()

    def _build_sample(
        self,
        action: np.ndarray,
        previous_command: Optional[np.ndarray],
        previous_valid_state: Optional[Dict[str, np.ndarray]],
        robot_interface: FrankaInterface,
        camera_subscribers: Dict[int, RedisFrameSubscriber],
        controller_cfg,
        skip_counters: Dict[str, int],
    ) -> Optional[Dict[str, object]]:
        if robot_interface.state_buffer_size == 0:
            skip_counters["missing_robot_state"] += 1
            return None
        if robot_interface.gripper_state_buffer_size == 0:
            skip_counters["missing_gripper_state"] += 1
            return None

        last_state = robot_interface.last_state
        last_gripper_state = robot_interface._gripper_state_buffer[-1]

        camera_frames = {}
        optional_camera_roles = set(self.task_cfg.collection.optional_camera_roles)
        for camera_cfg in self.task_cfg.cameras:
            frame = camera_subscribers[camera_cfg.camera_id].get_frame()
            is_optional = camera_cfg.role in optional_camera_roles
            if frame is None or frame.color is None:
                if camera_cfg.require_rgb and not is_optional:
                    skip_counters["missing_rgb"] += 1
                    return None
            if camera_cfg.require_depth and (frame is None or frame.depth is None) and not is_optional:
                skip_counters["missing_depth"] += 1
                return None
            if frame is None or frame.color is None:
                color = np.zeros(
                    (camera_cfg.resize.height, camera_cfg.resize.width, 3),
                    dtype=np.uint8,
                )
                depth = None
                if camera_cfg.enable_depth:
                    depth = np.zeros(
                        (camera_cfg.resize.height, camera_cfg.resize.width),
                        dtype=np.uint16,
                    )
                valid = False
                depth_valid = False
            else:
                color = frame.color
                depth = frame.depth
                depth_valid = depth is not None
                if depth is None and camera_cfg.enable_depth:
                    depth = np.zeros(
                        (camera_cfg.resize.height, camera_cfg.resize.width),
                        dtype=np.uint16,
                    )
                valid = True
            camera_frames[camera_cfg.camera_id] = {
                "color": color,
                "depth": depth,
                "calibration": camera_subscribers[camera_cfg.camera_id].get_calibration() or {},
                "valid": bool(valid),
                "depth_valid": bool(depth_valid),
            }

        ee_pose = np.array(last_state.O_T_EE, dtype=np.float64).reshape(4, 4).transpose()
        gripper_states = np.array([last_gripper_state.width], dtype=np.float64)

        if self.task_cfg.collection.state_zero_fallback:
            ee_pose = self._fallback_zero_state(
                value=ee_pose,
                key="ee_pose",
                previous_valid_state=previous_valid_state,
            )
            gripper_states = self._fallback_zero_state(
                value=gripper_states,
                key="gripper_states",
                previous_valid_state=previous_valid_state,
            )

        action_delta = self._compute_delta_action(
            action=action,
            previous_command=previous_command,
            controller_is_delta=bool(controller_cfg.is_delta),
        )

        return {
            # `delta_actions` is the single canonical teleoperation command stream.
            # For the default OSC controllers this is the exact controller-space
            # end-effector delta command sent to Franka/Deoxys after the
            # SpaceMouse input has already been mapped into Deoxys controller
            # convention. It is not the raw HID/SpaceMouse delta.
            # Non-delta controllers fall back to the difference between
            # consecutive controller commands so the dataset remains delta-centric.
            "delta_action": action_delta,
            "controller_command": np.array(action, dtype=np.float64),
            "ee_state_10d": np.concatenate(
                [pose_to_position_rotation6d(ee_pose), gripper_states]
            ).astype(np.float64),
            "camera_frames": camera_frames,
            "valid_state_cache": {
                "ee_pose": ee_pose.copy(),
                "gripper_states": gripper_states.copy(),
            },
        }

    @staticmethod
    def _compute_delta_action(
        action: np.ndarray,
        previous_command: Optional[np.ndarray],
        controller_is_delta: bool,
    ) -> np.ndarray:
        """Convert the current controller command into the canonical delta-action stream."""

        action_array = np.array(action, dtype=np.float64)
        if controller_is_delta:
            return action_array
        if previous_command is None:
            return np.zeros_like(action_array, dtype=np.float64)
        return action_array - np.array(previous_command, dtype=np.float64)

    def _apply_action_transform(self, action: np.ndarray) -> np.ndarray:
        """Apply an optional per-dimension action multiplier.

        This is the pipeline-native replacement for ad hoc "inverted SpaceMouse"
        branches from older scripts. A task can flip any action axis by setting a
        negative multiplier in the config.
        """

        multipliers = np.asarray(self.task_cfg.collection.action_multipliers, dtype=np.float64)
        if multipliers.shape != action.shape:
            logger.warning(
                "Ignoring action multipliers with mismatched shape %s for action shape %s",
                multipliers.shape,
                action.shape,
            )
            return action
        return action * multipliers

    def _fallback_zero_state(
        self,
        value: np.ndarray,
        key: str,
        previous_valid_state: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Replace obviously invalid all-zero state reads with the previous valid sample."""

        if np.sum(np.abs(value)) > self.task_cfg.collection.state_zero_threshold:
            return value
        if previous_valid_state is None or key not in previous_valid_state:
            return value
        logger.debug("Using previous valid %s because the current state read was near zero", key)
        return np.asarray(previous_valid_state[key], dtype=np.float64)

    def _finalize_run(
        self,
        run_dir: Path,
        manifest: Dict[str, object],
        data: Dict[str, List[np.ndarray]],
        camera_color: Dict[int, List[np.ndarray]],
        camera_depth: Dict[int, List[np.ndarray]],
        camera_depth_valid: Dict[int, List[np.ndarray]],
        camera_calibration: Dict[int, dict],
        camera_valid: Dict[int, List[np.ndarray]],
        skip_counters: Dict[str, int],
    ) -> str:
        sample_count = len(data["delta_action"])
        lengths = [sample_count]
        lengths.extend(len(values) for values in data.values())
        lengths.extend(len(values) for values in camera_color.values())
        lengths.extend(len(values) for values in camera_valid.values())
        lengths.extend(
            len(camera_depth[camera_cfg.camera_id])
            for camera_cfg in self.task_cfg.cameras
            if camera_cfg.enable_depth
        )
        lengths.extend(
            len(camera_depth_valid[camera_cfg.camera_id])
            for camera_cfg in self.task_cfg.cameras
            if camera_cfg.enable_depth
        )
        if sample_count == 0 or len(set(lengths)) != 1:
            manifest["status"] = "failed"
            manifest["failure_reason"] = "Accepted sample streams were empty or length-mismatched."
            manifest["num_samples"] = sample_count
            manifest["skip_counters"] = skip_counters
            self._write_manifest(run_dir, manifest)
            return "failed"

        np.savez_compressed(
            run_dir / "testing_demo_delta_action.npz",
            data=np.stack(data["delta_action"]),
        )
        np.savez_compressed(
            run_dir / "testing_demo_ee_state_10d.npz",
            data=np.stack(data["ee_state_10d"]),
        )

        camera_depth_scales = {}
        camera_calibration_files = {}
        for camera_cfg in self.task_cfg.cameras:
            camera_id = camera_cfg.camera_id
            np.savez_compressed(
                run_dir / f"testing_demo_camera_{camera_id}_color.npz",
                data=np.stack(camera_color[camera_id]),
            )
            np.savez_compressed(
                run_dir / f"testing_demo_camera_{camera_id}_valid.npz",
                data=np.stack(camera_valid[camera_id]),
            )
            if camera_cfg.enable_depth:
                np.savez_compressed(
                    run_dir / f"testing_demo_camera_{camera_id}_depth.npz",
                    data=np.stack(camera_depth[camera_id]),
                )
                np.savez_compressed(
                    run_dir / f"testing_demo_camera_{camera_id}_depth_valid.npz",
                    data=np.stack(camera_depth_valid[camera_id]),
                )
            calibration = camera_calibration.get(camera_id, {})
            depth_scale_value = float(calibration.get("depth_scale_m_per_unit", 0.0))
            if calibration:
                calibration_path = run_dir / f"testing_demo_camera_{camera_id}_calibration.json"
                calibration_path.write_text(
                    json.dumps(calibration, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                camera_calibration_files[camera_cfg.role] = str(calibration_path)
            camera_depth_scales[camera_cfg.role] = depth_scale_value

        manifest["status"] = "success"
        manifest["num_samples"] = sample_count
        manifest["camera_depth_scales"] = camera_depth_scales
        manifest["camera_calibration_files"] = camera_calibration_files
        manifest["skip_counters"] = skip_counters
        manifest["field_units"] = describe_field_units(
            [
                "delta_actions",
                "ee_state_10d",
                "valid",
                "depth_valid",
                "depth_scale_m_per_unit",
            ]
        )
        self._write_manifest(run_dir, manifest)
        return "success"

    def _discard_run(
        self,
        run_dir: Path,
        manifest: Dict[str, object],
        skip_counters: Dict[str, int],
        reason: str,
    ) -> str:
        """Discard a run instead of saving it, typically after manual interruption."""

        manifest["status"] = "discarded"
        manifest["failure_reason"] = reason
        manifest["num_samples"] = 0
        manifest["skip_counters"] = skip_counters
        shutil.rmtree(run_dir, ignore_errors=True)
        return "discarded"

    def _log_run_summary(
        self,
        status: str,
        num_samples: int,
        skip_counters: Dict[str, int],
    ) -> None:
        """Print a compact operator-facing summary after collection ends."""

        summary_lines = [
            "Collection summary:",
            f"  status: {status}",
            f"  valid_samples: {num_samples}",
            f"  missing_robot_state: {skip_counters['missing_robot_state']}",
            f"  missing_gripper_state: {skip_counters['missing_gripper_state']}",
            f"  missing_rgb: {skip_counters['missing_rgb']}",
            f"  missing_depth: {skip_counters['missing_depth']}",
            f"  no_motion_yet: {skip_counters['no_motion_yet']}",
            f"  invalid_action: {skip_counters['invalid_action']}",
            f"  spacemouse_finish_signal: {skip_counters['spacemouse_finish_signal']}",
        ]
        logger.info("\n%s", "\n".join(summary_lines))

    def _initial_manifest(self, run_dir: Path, preset: Optional[ResetPreset]) -> Dict[str, object]:
        controller_cfg_bytes = self.task_cfg.controller_cfg_path.read_bytes()
        return {
            "task": self.task_cfg.name,
            "task_config_stem": self.task_cfg.config_stem,
            "task_config_path": self.task_cfg.config_path,
            "run_dir": str(run_dir),
            "status": "recording",
            "preset": None if preset is None else preset.to_manifest(),
            "controller_type": self.task_cfg.controller_type,
            "controller_cfg_path": str(self.task_cfg.controller_cfg_path),
            "controller_cfg_hash": __import__("hashlib").sha256(controller_cfg_bytes).hexdigest(),
            "interface_cfg_path": str(self.task_cfg.interface_cfg_path),
            "default_reset_preset": self.task_cfg.default_reset_preset,
            "control_rates_hz": self.task_cfg.control_rates_hz,
            "camera_capture_rates_hz": self.task_cfg.camera_capture_rates_hz,
            "nominal_dataset_rate_hz": self.task_cfg.control_rates_hz.get("policy", 0),
            "camera_roles": self.task_cfg.camera_roles,
            "camera_ids": {
                camera.role: camera.camera_id for camera in self.task_cfg.cameras
            },
            "camera_serials": {
                camera.role: camera.serial_number for camera in self.task_cfg.cameras
            },
            "camera_configured_extrinsics": {
                camera.role: [
                    {
                        "name": extrinsic.name,
                        "reference_frame": extrinsic.reference_frame,
                        "target_frame": extrinsic.target_frame,
                        "transform_row_major": extrinsic.transform,
                        "convention": extrinsic.convention,
                    }
                    for extrinsic in camera.extrinsics
                ]
                for camera in self.task_cfg.cameras
            },
            "resize": {
                camera.role: {
                    "width_px": camera.resize.width,
                    "height_px": camera.resize.height,
                }
                for camera in self.task_cfg.cameras
            },
            "depth_enabled": {
                camera.role: camera.enable_depth for camera in self.task_cfg.cameras
            },
            "depth_required": {
                camera.role: camera.require_depth for camera in self.task_cfg.cameras
            },
            "optional_camera_roles": list(self.task_cfg.collection.optional_camera_roles),
            "spacemouse_vendor_id": self.task_cfg.spacemouse_vendor_id,
            "spacemouse_product_id": self.task_cfg.spacemouse_product_id,
            "timestamps": {"start_unix_sec": time.time()},
            "failure_reason": "",
            "manifest_comment": "Numeric units are defined in `field_units` and dataset attrs when available.",
            "action_semantics": "controller_delta_action",
            "action_semantics_detail": (
                "For OSC controllers this file stores the exact controller-space "
                "end-effector delta command sent to Franka/Deoxys after "
                "SpaceMouse-to-controller mapping. It does not store the raw "
                "SpaceMouse HID delta."
            ),
            "action_frame": "controller/end-effector command convention",
            "action_source_device": "SpaceMouse mapped into Deoxys controller convention",
            "raw_delta_action_file": "testing_demo_delta_action.npz",
            "hdf5_delta_action_dataset": "delta_actions",
            "ee_state_10d_source_field": "Derived from the measured O_T_EE pose matrix plus measured gripper width because Franka/Deoxys exposes pose matrices, not a native 10D learning state field.",
            "raw_run_contains": [
                "manifest.json",
                "testing_demo_delta_action.npz",
                "testing_demo_ee_state_10d.npz",
                "testing_demo_camera_<id>_color.npz",
                "testing_demo_camera_<id>_valid.npz",
                "testing_demo_camera_<id>_depth.npz (when depth is enabled)",
                "testing_demo_camera_<id>_depth_valid.npz (when depth is enabled)",
                "testing_demo_camera_<id>_calibration.json",
            ],
        }

    def _write_manifest(self, run_dir: Path, manifest: Dict[str, object]) -> None:
        manifest.setdefault("timestamps", {})
        manifest["timestamps"]["updated_unix_sec"] = time.time()
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
