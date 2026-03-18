#!/usr/bin/env python3
"""Capture calibration images, and optionally matching robot poses.

This helper is designed to work with the maintained `deoxys.data` camera path.
It subscribes to one Redis-published camera stream and lets the operator save
individual calibration snapshots with a key press.

If `--with-robot-pose` is enabled, each saved sample also stores the latest
measured Franka end-effector pose `O_T_EE` as a 4x4 row-major transform in
meters. That format is convenient for later hand-eye calibration where we need
`base_T_ee` / `world_T_ee` samples alongside wrist-camera images.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from deoxys.data.config import load_task_config
from deoxys.data.redis_io import RedisFrameSubscriber
from deoxys.franka_interface import FrankaInterface


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The `opencv-python` package is required to capture calibration samples."
        ) from exc
    return cv2


def _pose_matrix_from_state(robot_state) -> np.ndarray:
    """Return `O_T_EE` as a 4x4 row-major transform.

    Units:
    - translation entries: meters
    - rotation entries: unitless
    """

    return np.array(robot_state.O_T_EE, dtype=np.float64).reshape(4, 4).transpose()


def _overlay_lines(image_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    """Draw a compact operator HUD onto the preview image."""

    cv2 = _load_cv2()
    canvas = image_bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task config stem, e.g. fr3_dual_realsense")
    parser.add_argument("--camera-role", required=True, help="Configured camera role, e.g. agentview or wrist")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for captured PNGs and `samples.json`.",
    )
    parser.add_argument(
        "--with-robot-pose",
        action="store_true",
        help="Also save the latest measured `O_T_EE` pose for each capture.",
    )
    args = parser.parse_args()

    cv2 = _load_cv2()
    task_cfg = load_task_config(args.task)
    camera_cfg = next((camera for camera in task_cfg.cameras if camera.role == args.camera_role), None)
    if camera_cfg is None:
        raise ValueError(
            f"Camera role `{args.camera_role}` is not defined in task `{args.task}`."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.json"

    subscriber = RedisFrameSubscriber(task_cfg.redis, camera_cfg)
    robot_interface = None
    if args.with_robot_pose:
        robot_interface = FrankaInterface(str(task_cfg.interface_cfg_path), use_visualizer=False)

    samples: List[Dict[str, object]] = []
    if samples_path.exists():
        payload = json.loads(samples_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            samples = list(payload["samples"])

    try:
        while True:
            frame = subscriber.get_frame()
            if frame is None or frame.color is None:
                time.sleep(0.03)
                continue

            preview = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
            lines = [
                f"role: {camera_cfg.role}",
                f"saved samples: {len(samples)}",
                "SPACE: capture sample",
                "Q / ESC: quit",
            ]
            if args.with_robot_pose:
                has_state = robot_interface is not None and robot_interface.state_buffer_size > 0
                lines.append(f"robot pose ready: {'yes' if has_state else 'no'}")
            preview = _overlay_lines(preview, lines)
            cv2.imshow("deoxys calibration capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key != 32:
                continue

            robot_pose_row_major = None
            if args.with_robot_pose:
                if robot_interface is None or robot_interface.state_buffer_size == 0:
                    print("Skipping capture because no robot state is available yet.")
                    continue
                robot_pose_row_major = _pose_matrix_from_state(robot_interface.last_state).reshape(-1).tolist()

            sample_index = len(samples)
            image_path = output_dir / f"{camera_cfg.role}_{sample_index:04d}.png"
            ok = cv2.imwrite(str(image_path), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write calibration image: {image_path}")

            sample_entry: Dict[str, object] = {
                "image": str(image_path),
                "camera_role": camera_cfg.role,
                "camera_id": camera_cfg.camera_id,
                "capture_unix_sec": time.time(),
                "frame_id": int(frame.info.get("frame_id", -1)),
                "publish_timestamp_sec": float(frame.info.get("publish_timestamp_sec", float("nan"))),
                "acquisition_timestamp_ms": float(frame.info.get("acquisition_timestamp_ms", float("nan"))),
            }
            if robot_pose_row_major is not None:
                sample_entry["base_T_ee"] = robot_pose_row_major

            samples.append(sample_entry)
            samples_path.write_text(
                json.dumps(
                    {
                        "task": task_cfg.name,
                        "task_config_stem": task_cfg.config_stem,
                        "camera_role": camera_cfg.role,
                        "with_robot_pose": bool(args.with_robot_pose),
                        "samples": samples,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(f"Captured sample {sample_index} -> {image_path}")
    finally:
        cv2.destroyAllWindows()
        if robot_interface is not None:
            robot_interface.close()

    print(f"Wrote sample manifest: {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
