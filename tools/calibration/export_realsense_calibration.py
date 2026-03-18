#!/usr/bin/env python3
"""Export RealSense calibration metadata and write an extrinsics template.

This script does two practical jobs:

1. Query the connected RealSense cameras and save the calibration data that the
   device already knows by itself:
   - color intrinsics
   - depth intrinsics
   - resized intrinsics that match the dataset image size
   - depth-to-color and color-to-depth sensor extrinsics
   - depth scale

2. Write a task-facing YAML template for setup-specific extrinsics that the
   camera does *not* know automatically, such as:
   - world_T_agentview
   - robot_base_T_agentview
   - ee_T_wrist_camera

Important:
The RealSense device can provide its own internal calibration, but it cannot
magically know where it sits relative to your robot base, world frame, or end
effector. Those transforms must come from your own calibration workflow.

Typical ways to obtain those missing matrices:
- External / static camera:
  Place a checkerboard or AprilTag board in a known robot/world frame and solve
  the camera pose using OpenCV `solvePnP`, Kalibr, or an AprilTag pose stack.
- Wrist / hand camera:
  Run a hand-eye calibration procedure (`AX = XB`) using multiple robot poses
  and the observed target poses. Common tools include MoveIt calibration,
  `easy_handeye`, or your own OpenCV-based hand-eye script.

The output of this script gives you a clean place to paste those final matrices
later, instead of hunting through the pipeline code again.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from deoxys.data.config import load_task_config


def _load_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "The `pyrealsense2` package is required to export RealSense calibration."
        ) from exc
    return rs


def _serialize_intrinsics(intrinsics) -> dict:
    """Convert a RealSense intrinsics object into JSON/YAML-friendly values.

    Units:
    - width_px / height_px: pixels
    - fx_px / fy_px: focal length in pixels
    - ppx_px / ppy_px: principal point in pixels
    - distortion_coeffs: unitless
    """

    return {
        "width_px": int(intrinsics.width),
        "height_px": int(intrinsics.height),
        "fx_px": float(intrinsics.fx),
        "fy_px": float(intrinsics.fy),
        "ppx_px": float(intrinsics.ppx),
        "ppy_px": float(intrinsics.ppy),
        "model_id": int(intrinsics.model),
        "model_name": str(intrinsics.model),
        "distortion_coeffs": [float(value) for value in intrinsics.coeffs],
    }


def _resize_intrinsics(native_intrinsics: dict, width_px: int, height_px: int) -> dict:
    """Scale native intrinsics to the resized dataset image size.

    This is useful because the collector stores resized images. If later code
    consumes the resized frames directly, it should use these scaled intrinsics
    rather than the native stream values.
    """

    width_scale = float(width_px) / float(native_intrinsics["width_px"])
    height_scale = float(height_px) / float(native_intrinsics["height_px"])
    resized = dict(native_intrinsics)
    resized["width_px"] = int(width_px)
    resized["height_px"] = int(height_px)
    resized["fx_px"] = float(native_intrinsics["fx_px"]) * width_scale
    resized["fy_px"] = float(native_intrinsics["fy_px"]) * height_scale
    resized["ppx_px"] = float(native_intrinsics["ppx_px"]) * width_scale
    resized["ppy_px"] = float(native_intrinsics["ppy_px"]) * height_scale
    return resized


def _serialize_rs_extrinsics(extrinsics) -> dict:
    """Serialize RealSense sensor-to-sensor extrinsics.

    Units:
    - rotation_row_major: unitless rotation matrix entries
    - translation_m: meters
    """

    return {
        "rotation_row_major": [float(value) for value in extrinsics.rotation],
        "translation_m": [float(value) for value in extrinsics.translation],
        "convention": "3x3 row-major rotation plus translation in meters",
    }


def _device_report(camera_cfg) -> dict:
    """Collect calibration metadata for one task camera using its configured serial."""

    rs = _load_realsense()
    pipeline = rs.pipeline()
    config = rs.config()
    if camera_cfg.serial_number:
        config.enable_device(camera_cfg.serial_number)
    config.enable_stream(
        rs.stream.color,
        camera_cfg.width,
        camera_cfg.height,
        rs.format.bgr8,
        camera_cfg.fps,
    )
    if camera_cfg.enable_depth:
        config.enable_stream(
            rs.stream.depth,
            camera_cfg.width,
            camera_cfg.height,
            rs.format.z16,
            camera_cfg.fps,
        )

    profile = pipeline.start(config)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_native = _serialize_intrinsics(color_profile.get_intrinsics())
        color_resized = _resize_intrinsics(
            color_native,
            camera_cfg.resize.width,
            camera_cfg.resize.height,
        )
        report = {
            "role": camera_cfg.role,
            "camera_id": camera_cfg.camera_id,
            "serial_number": camera_cfg.serial_number,
            "rgb_stream": {
                "native_intrinsics": color_native,
                "resized_intrinsics": color_resized,
            },
            "configured_task_extrinsics": [
                {
                    "name": extrinsic.name,
                    "reference_frame": extrinsic.reference_frame,
                    "target_frame": extrinsic.target_frame,
                    "transform_row_major": extrinsic.transform,
                    "convention": extrinsic.convention,
                }
                for extrinsic in camera_cfg.extrinsics
            ],
        }

        if camera_cfg.enable_depth:
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_native = _serialize_intrinsics(depth_profile.get_intrinsics())
            depth_resized = _resize_intrinsics(
                depth_native,
                camera_cfg.resize.width,
                camera_cfg.resize.height,
            )
            report["depth_stream"] = {
                "native_intrinsics": depth_native,
                "resized_intrinsics": depth_resized,
                "depth_scale_m_per_unit": float(depth_sensor.get_depth_scale()),
            }
            report["device_extrinsics"] = {
                "depth_to_color": _serialize_rs_extrinsics(
                    depth_profile.get_extrinsics_to(color_profile)
                ),
                "color_to_depth": _serialize_rs_extrinsics(
                    color_profile.get_extrinsics_to(depth_profile)
                ),
            }

        return report
    finally:
        pipeline.stop()


def _write_task_extrinsics_template(task_cfg: TaskConfig, output_dir: Path) -> Path:
    """Write a human-editable YAML template for the missing setup extrinsics.

    This file is intentionally a template. The numbers are left blank because
    fake identity transforms are worse than missing transforms in a dataset.
    """

    output_path = output_dir / f"{task_cfg.name}_extrinsics_template.yml"
    lines = [
        "# Task extrinsics template",
        "# Fill these transforms only after you run a real calibration.",
        "# Each transform must be a 4x4 homogeneous matrix in row-major order.",
        "# Translation units are meters. Rotation is unitless.",
        "",
        f"task: {task_cfg.name}",
        "cameras:",
    ]
    for camera_cfg in task_cfg.cameras:
        lines.extend(
            [
                f"  - role: {camera_cfg.role}",
                f"    serial_number: \"{camera_cfg.serial_number}\"",
                "    extrinsics:",
                "      # Example names you might use:",
                "      # - world_T_camera",
                "      # - robot_base_T_camera",
                "      # - ee_T_wrist_camera",
                "      - name: FILL_ME",
                "        reference_frame: FILL_ME",
                "        target_frame: FILL_ME",
                "        transform: []",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task config stem, e.g. fr3_dual_realsense")
    parser.add_argument(
        "--output-dir",
        default="tools/calibration/output",
        help="Directory for calibration reports and templates.",
    )
    args = parser.parse_args()

    task_cfg = load_task_config(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "task": task_cfg.name,
        "generated_at_local": datetime.now().isoformat(),
        "cameras": [_device_report(camera_cfg) for camera_cfg in task_cfg.cameras],
    }
    report_path = output_dir / f"{task_cfg.name}_realsense_calibration_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    template_path = _write_task_extrinsics_template(task_cfg, output_dir)
    print(f"Wrote RealSense calibration report: {report_path}")
    print(f"Wrote task extrinsics template: {template_path}")
    print("")
    print("Next steps:")
    print("1. Keep the RealSense report; it contains device-native intrinsics and sensor extrinsics.")
    print("2. Run your world/robot/hand-eye calibration separately.")
    print("3. Paste the resulting 4x4 setup extrinsics into the task YAML.")
    print("4. Re-run collection so the calibration lands in raw runs and HDF5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
