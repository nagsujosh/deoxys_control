"""Process management for Redis and camera nodes."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import TaskConfig
from .logging_utils import get_data_logger
from .redis_io import RedisFrameSubscriber, make_redis_client

logger = get_data_logger("process_manager")


@dataclass
class ManagedProcess:
    """Tracks one child process started by the pipeline."""

    name: str
    process: subprocess.Popen


class ProcessManager:
    """Start and supervise Redis and camera-node processes."""

    def __init__(self, task_cfg: TaskConfig):
        self.task_cfg = task_cfg
        self._processes: Dict[str, ManagedProcess] = {}

    def start_redis(self) -> None:
        """Start Redis if the task owns that process."""

        if not self.task_cfg.redis.start_managed:
            logger.info("Using externally managed Redis at %s:%s", self.task_cfg.redis.host, self.task_cfg.redis.port)
            return
        if "redis" in self._processes:
            return
        logger.info("Starting managed Redis with command: %s", " ".join(self.task_cfg.redis.command))
        process = subprocess.Popen(  # noqa: S603
            self.task_cfg.redis.command,
        )
        self._processes["redis"] = ManagedProcess(name="redis", process=process)
        try:
            self._wait_for_process_start("redis")
            self._wait_for_redis()
            # Re-check the specific managed child after Redis becomes reachable so an
            # already-running external Redis cannot mask a dead managed process.
            self._wait_for_process_start("redis", grace_sec=1.0)
        except Exception:
            self._cleanup_process("redis")
            raise
        logger.info("Managed Redis is healthy")

    def start_camera_nodes(self) -> None:
        """Start one camera-node process per configured camera role."""

        for camera_cfg in self.task_cfg.cameras:
            process_name = f"camera:{camera_cfg.role}"
            if process_name in self._processes:
                continue
            logger.info(
                "Starting camera node for role=%s serial=%s",
                camera_cfg.role,
                camera_cfg.serial_number or "<auto>",
            )
            process = subprocess.Popen(  # noqa: S603
                (
                    [
                        sys.executable,
                        "-m",
                        "deoxys.data.camera_node",
                        "--config",
                        self.task_cfg.config_path,
                        "--role",
                        camera_cfg.role,
                    ]
                    if self.task_cfg.config_path
                    else [
                        sys.executable,
                        "-m",
                        "deoxys.data.camera_node",
                        "--task",
                        self.task_cfg.config_stem or self.task_cfg.name,
                        "--role",
                        camera_cfg.role,
                    ]
                ),
            )
            self._processes[process_name] = ManagedProcess(
                name=process_name, process=process
            )
            try:
                self._wait_for_process_start(process_name)
            except Exception:
                self._cleanup_process(process_name)
                raise

    def stop_all(self) -> None:
        """Terminate all managed child processes."""

        for managed in reversed(list(self._processes.values())):
            if managed.process.poll() is None:
                logger.info("Stopping managed process `%s` (pid=%s)", managed.name, managed.process.pid)
                managed.process.terminate()
                try:
                    managed.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing unresponsive managed process `%s`", managed.name)
                    managed.process.kill()
        self._processes.clear()

    def health_report(self) -> Dict[str, Dict[str, object]]:
        """Report process liveness and camera freshness."""

        report: Dict[str, Dict[str, object]] = {}
        report["redis"] = {"reachable": self._redis_ping()}
        now = time.time()
        for camera_cfg in self.task_cfg.cameras:
            process_name = f"camera:{camera_cfg.role}"
            frame = None
            try:
                subscriber = RedisFrameSubscriber(self.task_cfg.redis, camera_cfg)
                frame = subscriber.get_frame()
            except Exception:
                frame = None
            is_fresh = False
            age_sec = None
            if frame is not None:
                publish_timestamp = frame.info.get("publish_timestamp_sec", float("nan"))
                try:
                    publish_timestamp = float(publish_timestamp)
                except (TypeError, ValueError):
                    publish_timestamp = float("nan")
                if publish_timestamp == publish_timestamp:
                    age_sec = now - publish_timestamp
                    is_fresh = age_sec <= self.task_cfg.redis.freshness_timeout_sec
            report[camera_cfg.role] = {
                "fresh": is_fresh,
                "age_sec": age_sec,
                "frame_id": None if frame is None else frame.info.get("frame_id"),
                # This field is only knowable for processes started by this
                # ProcessManager instance. Fresh Redis frames remain the
                # source-of-truth for cross-process health checks such as
                # `deoxys.data preflight`.
                "managed_process_alive": (
                    None
                    if process_name not in self._processes
                    else self._processes[process_name].process.poll() is None
                ),
            }
        return report

    def _redis_ping(self) -> bool:
        try:
            return bool(make_redis_client(self.task_cfg.redis).ping())
        except Exception:
            return False

    def _wait_for_redis(self) -> None:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._redis_ping():
                return
            time.sleep(0.1)
        raise RuntimeError("Timed out waiting for managed Redis to become healthy.")

    def _wait_for_process_start(self, process_name: str, grace_sec: float = 0.5) -> None:
        """Fail fast if a newly spawned managed process exits immediately."""

        managed = self._processes[process_name]
        deadline = time.time() + grace_sec
        while time.time() < deadline:
            if managed.process.poll() is not None:
                raise RuntimeError(
                    f"Managed process `{process_name}` exited early with code {managed.process.returncode}."
                )
            time.sleep(0.05)

    def _cleanup_process(self, process_name: str) -> None:
        """Drop a failed managed process entry and terminate it if needed."""

        managed = self._processes.pop(process_name, None)
        if managed is None:
            return
        if managed.process.poll() is None:
            managed.process.terminate()
            try:
                managed.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                managed.process.kill()
