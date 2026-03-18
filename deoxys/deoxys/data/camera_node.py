"""Redis-backed RealSense camera publisher."""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from .config import CameraConfig, load_task_config, load_task_config_from_path
from .logging_utils import get_data_logger
from .redis_io import RedisFramePublisher

logger = get_data_logger("camera_node")


def _load_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "The `pyrealsense2` package is required to run camera nodes."
        ) from exc
    return rs


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The `opencv-python` package is required to run camera nodes."
        ) from exc
    return cv2


class RealSenseCameraNode:
    """Capture RGB-D frames and publish them to Redis."""

    def __init__(self, task_name: str | None, role: str, config_path: str | None = None):
        self.task_cfg = (
            load_task_config_from_path(config_path)
            if config_path
            else load_task_config(str(task_name))
        )
        self.camera_cfg = next(
            camera for camera in self.task_cfg.cameras if camera.role == role
        )
        self.publisher = RedisFramePublisher(self.task_cfg.redis, self.camera_cfg)
        self.rs = _load_realsense()
        self.pipeline = self.rs.pipeline()
        self.align = self.rs.align(self.rs.stream.color)
        self._calibration_publish_interval_sec = 2.0
        self._last_calibration_publish_time_sec = 0.0

    @staticmethod
    def _serialize_intrinsics(intrinsics) -> dict:
        """Serialize RealSense intrinsics into JSON-friendly metadata."""

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
            "units_comment": "Focal length and principal point are in pixels. Distortion coefficients are unitless.",
        }

    @staticmethod
    def _resize_intrinsics(intrinsics_payload: dict, target_width: int, target_height: int) -> dict:
        """Scale native intrinsics to the resized output image dimensions."""

        width_scale = float(target_width) / float(intrinsics_payload["width_px"])
        height_scale = float(target_height) / float(intrinsics_payload["height_px"])
        resized = dict(intrinsics_payload)
        resized["width_px"] = int(target_width)
        resized["height_px"] = int(target_height)
        resized["fx_px"] = float(intrinsics_payload["fx_px"]) * width_scale
        resized["fy_px"] = float(intrinsics_payload["fy_px"]) * height_scale
        resized["ppx_px"] = float(intrinsics_payload["ppx_px"]) * width_scale
        resized["ppy_px"] = float(intrinsics_payload["ppy_px"]) * height_scale
        resized["resize_comment"] = (
            "Scaled from the native stream intrinsics to the resized output image dimensions."
        )
        return resized

    @staticmethod
    def _serialize_rs_extrinsics(extrinsics) -> dict:
        """Serialize RealSense sensor-to-sensor extrinsics."""

        return {
            "rotation_row_major": [float(value) for value in extrinsics.rotation],
            "translation_m": [float(value) for value in extrinsics.translation],
            "convention": "3x3 row-major rotation plus translation in meters",
        }

    def _sensor_name(self, sensor) -> str:
        """Return a readable RealSense sensor name when available."""

        try:
            return str(sensor.get_info(self.rs.camera_info.name))
        except Exception:
            return "unknown"

    def _read_sensor_option(self, sensor, option_name: str) -> Optional[dict]:
        """Read one RealSense sensor option if supported.

        Values come back in the units used by librealsense for that option.
        For example, exposure is usually in microseconds and gain is sensor-specific.
        """

        option = getattr(self.rs.option, option_name, None)
        if option is None or not sensor.supports(option):
            return None
        try:
            value = sensor.get_option(option)
            payload = {"value": float(value)}
            try:
                payload["description"] = sensor.get_option_value_description(option, value)
            except Exception:
                pass
            return payload
        except Exception:
            return None

    def _serialize_sensor_settings(self, profile) -> dict:
        """Capture camera acquisition settings that affect image statistics."""

        device = profile.get_device()
        sensors = list(device.query_sensors())
        depth_sensor = device.first_depth_sensor() if self.camera_cfg.enable_depth else None
        color_sensor = None
        for sensor in sensors:
            sensor_name = self._sensor_name(sensor).lower()
            if "rgb" in sensor_name or "color" in sensor_name:
                color_sensor = sensor
                break
        if color_sensor is None and sensors:
            color_sensor = sensors[0]

        settings = {}
        if color_sensor is not None:
            settings["color_sensor"] = {
                "sensor_name": self._sensor_name(color_sensor),
                "exposure": self._read_sensor_option(color_sensor, "exposure"),
                "gain": self._read_sensor_option(color_sensor, "gain"),
                "white_balance": self._read_sensor_option(color_sensor, "white_balance"),
                "enable_auto_exposure": self._read_sensor_option(
                    color_sensor, "enable_auto_exposure"
                ),
                "enable_auto_white_balance": self._read_sensor_option(
                    color_sensor, "enable_auto_white_balance"
                ),
            }
        if depth_sensor is not None:
            settings["depth_sensor"] = {
                "sensor_name": self._sensor_name(depth_sensor),
                "exposure": self._read_sensor_option(depth_sensor, "exposure"),
                "gain": self._read_sensor_option(depth_sensor, "gain"),
                "enable_auto_exposure": self._read_sensor_option(
                    depth_sensor, "enable_auto_exposure"
                ),
                "emitter_enabled": self._read_sensor_option(depth_sensor, "emitter_enabled"),
                "laser_power": self._read_sensor_option(depth_sensor, "laser_power"),
                "visual_preset": self._read_sensor_option(depth_sensor, "visual_preset"),
            }
        settings["units_comment"] = (
            "Option values use librealsense/native sensor units. Exposure is typically in microseconds; "
            "gain and presets are device-specific."
        )
        return settings

    def _build_calibration_payload(self, profile, depth_scale: Optional[float]) -> dict:
        """Build one stable per-camera calibration payload."""

        color_profile = profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        color_native = self._serialize_intrinsics(color_profile.get_intrinsics())
        color_resized = self._resize_intrinsics(
            color_native,
            self.camera_cfg.resize.width,
            self.camera_cfg.resize.height,
        )

        calibration = {
            "role": self.camera_cfg.role,
            "camera_id": self.camera_cfg.camera_id,
            "serial_number": self.camera_cfg.serial_number,
            "color_intrinsics": {
                "native": color_native,
                "resized": color_resized,
            },
            "configured_extrinsics": [
                {
                    "name": extrinsic.name,
                    "reference_frame": extrinsic.reference_frame,
                    "target_frame": extrinsic.target_frame,
                    "transform_row_major": extrinsic.transform,
                    "convention": extrinsic.convention,
                    "units_comment": "Translation is in meters. Rotation is unitless.",
                }
                for extrinsic in self.camera_cfg.extrinsics
            ],
            "acquisition_settings": self._serialize_sensor_settings(profile),
        }

        if self.camera_cfg.enable_depth:
            depth_profile = profile.get_stream(self.rs.stream.depth).as_video_stream_profile()
            depth_native = self._serialize_intrinsics(depth_profile.get_intrinsics())
            depth_resized = self._resize_intrinsics(
                depth_native,
                self.camera_cfg.resize.width,
                self.camera_cfg.resize.height,
            )
            calibration["depth_intrinsics"] = {
                "native": depth_native,
                "resized": depth_resized,
            }
            calibration["depth_scale_m_per_unit"] = float(depth_scale or 0.0)
            calibration["device_extrinsics"] = {
                "depth_to_color": self._serialize_rs_extrinsics(
                    depth_profile.get_extrinsics_to(color_profile)
                ),
                "color_to_depth": self._serialize_rs_extrinsics(
                    color_profile.get_extrinsics_to(depth_profile)
                ),
            }

        return calibration

    def _publish_calibration_if_due(self, calibration: dict, force: bool = False) -> None:
        """Republish calibration periodically so Redis restarts do not lose it."""

        now = time.time()
        if (
            force
            or self._last_calibration_publish_time_sec <= 0.0
            or now - self._last_calibration_publish_time_sec
            >= self._calibration_publish_interval_sec
        ):
            self.publisher.publish_calibration(calibration)
            self._last_calibration_publish_time_sec = now

    def run(self) -> None:
        """Run until interrupted."""

        cv2 = _load_cv2()
        logger.info(
            "Starting RealSense publisher for role=%s serial=%s rgb=%dx%d@%d resize=%dx%d depth=%s",
            self.camera_cfg.role,
            self.camera_cfg.serial_number or "<auto>",
            self.camera_cfg.width,
            self.camera_cfg.height,
            self.camera_cfg.fps,
            self.camera_cfg.resize.width,
            self.camera_cfg.resize.height,
            self.camera_cfg.enable_depth,
        )
        config = self.rs.config()
        if self.camera_cfg.serial_number:
            config.enable_device(self.camera_cfg.serial_number)
        config.enable_stream(
            self.rs.stream.color,
            self.camera_cfg.width,
            self.camera_cfg.height,
            self.rs.format.bgr8,
            self.camera_cfg.fps,
        )
        if self.camera_cfg.enable_depth:
            config.enable_stream(
                self.rs.stream.depth,
                self.camera_cfg.width,
                self.camera_cfg.height,
                self.rs.format.z16,
                self.camera_cfg.fps,
            )

        profile = self.pipeline.start(config)
        depth_scale = None
        if self.camera_cfg.enable_depth:
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            logger.info(
                "Depth enabled for role=%s with scale=%s meters/unit",
                self.camera_cfg.role,
                depth_scale,
            )
        calibration = self._build_calibration_payload(profile, depth_scale)
        self._publish_calibration_if_due(calibration, force=True)
        logger.info(
            "Published calibration metadata for role=%s with %s configured extrinsics",
            self.camera_cfg.role,
            len(calibration.get("configured_extrinsics", [])),
        )

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame() if self.camera_cfg.enable_depth else None

                if not color_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                if (
                    color.shape[1] != self.camera_cfg.resize.width
                    or color.shape[0] != self.camera_cfg.resize.height
                ):
                    color = cv2.resize(
                        color,
                        (self.camera_cfg.resize.width, self.camera_cfg.resize.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

                depth = None
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data())
                    if (
                        depth.shape[1] != self.camera_cfg.resize.width
                        or depth.shape[0] != self.camera_cfg.resize.height
                    ):
                        depth = cv2.resize(
                            depth,
                            (self.camera_cfg.resize.width, self.camera_cfg.resize.height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                frame_id = color_frame.get_frame_number()
                info = {
                    "role": self.camera_cfg.role,
                    "camera_id": self.camera_cfg.camera_id,
                    "serial_number": self.camera_cfg.serial_number,
                    "frame_id": int(frame_id),
                    "acquisition_timestamp_ms": float(color_frame.get_timestamp()),
                    "width": int(self.camera_cfg.resize.width),
                    "height": int(self.camera_cfg.resize.height),
                    "color_encoding": self.camera_cfg.color_encoding,
                    "color_space": "rgb",
                    "depth_encoding": self.camera_cfg.depth_encoding,
                    "has_depth": bool(depth is not None),
                    "depth_scale_m_per_unit": float(depth_scale or 0.0),
                }
                self._publish_calibration_if_due(calibration)
                self.publisher.publish(info=info, color_image=color, depth_image=depth)
        except KeyboardInterrupt:
            logger.info("Stopping camera publisher for role=%s", self.camera_cfg.role)
        finally:
            logger.info("RealSense pipeline stopped for role=%s", self.camera_cfg.role)
            self.pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--config")
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    if not args.task and not args.config:
        parser.error("One of --task or --config is required.")
    RealSenseCameraNode(task_name=args.task, role=args.role, config_path=args.config).run()


if __name__ == "__main__":
    main()
