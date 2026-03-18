"""Redis transport helpers for RGB-D camera frames."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import CameraConfig, RedisConfig
from .logging_utils import get_data_logger

logger = get_data_logger("redis_io")


def _load_redis():
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "The `redis` Python package is required for the data pipeline. "
            "Install the project requirements in the active environment."
        ) from exc
    return redis


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The `opencv-python` package is required for camera transport and viewing."
        ) from exc
    return cv2


def make_redis_client(redis_cfg: RedisConfig):
    redis = _load_redis()
    return redis.Redis(
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.db,
        decode_responses=False,
    )


@dataclass
class CameraFrame:
    """One decoded RGB-D frame plus metadata."""

    info: Dict[str, Any]
    color: Optional[np.ndarray]
    depth: Optional[np.ndarray]


class RedisFramePublisher:
    """Publish the latest RGB-D frame for one camera role."""

    def __init__(self, redis_cfg: RedisConfig, camera_cfg: CameraConfig):
        self.redis = make_redis_client(redis_cfg)
        self.camera_cfg = camera_cfg
        self.info_key = f"{camera_cfg.redis_namespace}:img_info"
        self.buffer_key = f"{camera_cfg.redis_namespace}:img_buffer"
        self.calibration_key = f"{camera_cfg.redis_namespace}:calibration"

    def publish(
        self,
        info: Dict[str, Any],
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
    ) -> None:
        """Serialize and publish the latest RGB-D frame snapshot.

        `color_image` is expected to be RGB in memory. The transport temporarily
        converts it to BGR only for OpenCV JPEG encoding, then subscribers
        decode and convert back to RGB so the rest of the pipeline remains
        explicitly RGB-D rather than BGR-D.
        """

        cv2 = _load_cv2()
        color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        success, color_buffer = cv2.imencode(".jpg", color_bgr)
        if not success:
            raise RuntimeError(f"Failed to encode RGB frame for {self.camera_cfg.role}")

        payload = {"color": color_buffer.tobytes()}
        if depth_image is not None:
            depth_success, depth_buffer = cv2.imencode(".png", depth_image)
            if not depth_success:
                raise RuntimeError(
                    f"Failed to encode depth frame for {self.camera_cfg.role}"
                )
            payload["depth"] = depth_buffer.tobytes()

        info = dict(info)
        info["publish_timestamp_sec"] = time.time()
        info.setdefault("color_space", "rgb")
        pipe = self.redis.pipeline()
        pipe.set(self.info_key, json.dumps(info).encode("utf-8"))
        pipe.set(self.buffer_key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        pipe.execute()

    def publish_calibration(self, calibration: Dict[str, Any]) -> None:
        """Publish stable calibration metadata for one camera role."""

        self.redis.set(
            self.calibration_key,
            json.dumps(calibration).encode("utf-8"),
        )


class RedisFrameSubscriber:
    """Read the latest RGB-D frame for one camera role."""

    def __init__(self, redis_cfg: RedisConfig, camera_cfg: CameraConfig):
        self.redis = make_redis_client(redis_cfg)
        self.camera_cfg = camera_cfg
        self.info_key = f"{camera_cfg.redis_namespace}:img_info"
        self.buffer_key = f"{camera_cfg.redis_namespace}:img_buffer"
        self.calibration_key = f"{camera_cfg.redis_namespace}:calibration"

    def get_frame(self) -> Optional[CameraFrame]:
        """Return the latest decoded frame, or None if no frame is available."""

        cv2 = _load_cv2()
        info_payload, buffer_payload = self.redis.mget([self.info_key, self.buffer_key])
        if not info_payload or not buffer_payload:
            return None

        info = json.loads(info_payload.decode("utf-8"))
        buffers = pickle.loads(buffer_payload)

        color = cv2.imdecode(
            np.frombuffer(buffers["color"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if color is not None:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        depth = None
        if "depth" in buffers:
            depth = cv2.imdecode(
                np.frombuffer(buffers["depth"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )

        return CameraFrame(info=info, color=color, depth=depth)

    def get_calibration(self) -> Optional[Dict[str, Any]]:
        """Return the latest published calibration payload, if any."""

        calibration_payload = self.redis.get(self.calibration_key)
        if not calibration_payload:
            return None
        return json.loads(calibration_payload.decode("utf-8"))

    def ping(self) -> bool:
        """Return whether Redis is reachable."""

        return bool(self.redis.ping())
