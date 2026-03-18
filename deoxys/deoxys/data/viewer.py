"""OpenCV-based Redis stream viewer."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import numpy as np

from .config import TaskConfig
from .logging_utils import get_data_logger
from .redis_io import RedisFrameSubscriber

logger = get_data_logger("viewer")
_VIEWER_WINDOW_NAME = "deoxys.data viewer"


def _configure_qt_fontdir() -> None:
    """Best-effort Qt font setup to avoid noisy OpenCV/Qt font warnings."""

    if os.environ.get("QT_QPA_FONTDIR") and os.environ.get("OPENCV_QT_FONTDIR"):
        return
    # Suppress noisy Qt warning spam from the OpenCV HighGUI backend. This is
    # display-only behavior and should not affect captured data.
    os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        if candidate.exists():
            os.environ.setdefault("QT_QPA_FONTDIR", str(candidate))
            os.environ.setdefault("OPENCV_QT_FONTDIR", str(candidate))
            return


def _load_cv2():
    _configure_qt_fontdir()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The `opencv-python` package is required to view live camera streams."
        ) from exc
    return cv2


def _overlay_status(image: np.ndarray, lines: List[str]) -> np.ndarray:
    cv2 = _load_cv2()
    display = image.copy()
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (10, 24 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return display


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    cv2 = _load_cv2()
    if depth is None:
        return None
    depth_float = depth.astype(np.float32)
    if np.max(depth_float) <= 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    normalized = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _safe_frame_age_sec(frame_info: dict) -> str:
    """Return a human-readable frame age string for overlays."""

    publish_timestamp = frame_info.get("publish_timestamp_sec", float("nan"))
    try:
        publish_timestamp = float(publish_timestamp)
    except (TypeError, ValueError):
        publish_timestamp = float("nan")
    if not np.isfinite(publish_timestamp):
        return "unknown"
    return f"{time.time() - publish_timestamp:.3f}"


def _screen_size_px() -> tuple[int, int]:
    """Best-effort screen size for display-only preview scaling."""

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1600, 900


def _fit_canvas_for_display(canvas: np.ndarray, margin_px: int = 120) -> np.ndarray:
    """Downscale a preview canvas to fit on screen without changing saved data."""

    cv2 = _load_cv2()
    screen_width, screen_height = _screen_size_px()
    max_width = max(320, screen_width - margin_px)
    max_height = max(240, screen_height - margin_px)
    canvas_height, canvas_width = canvas.shape[:2]

    scale = min(
        1.0,
        float(max_width) / float(canvas_width),
        float(max_height) / float(canvas_height),
    )
    if scale >= 1.0:
        return canvas

    resized_width = max(1, int(round(canvas_width * scale)))
    resized_height = max(1, int(round(canvas_height * scale)))
    return cv2.resize(
        canvas,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def _stack_panels_vertically(panels: List[np.ndarray]) -> np.ndarray:
    """Pad panels to a common width before vertical stacking."""

    if not panels:
        raise ValueError("At least one panel is required to build a canvas")
    max_width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[1] == max_width:
            padded.append(panel)
            continue
        pad_width = max_width - panel.shape[1]
        padded.append(
            np.pad(panel, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
        )
    return padded[0] if len(padded) == 1 else np.vstack(padded)


def _viewer_window_closed(cv2) -> bool:
    """Return whether the OpenCV preview window was closed by the window manager."""

    try:
        visible = cv2.getWindowProperty(_VIEWER_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        autosize = cv2.getWindowProperty(_VIEWER_WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)
    except Exception:
        return True
    return visible < 1 or autosize < 0


def run_viewer(task_cfg: TaskConfig, camera_roles: List[str]) -> None:
    """Display one or more live camera streams."""

    cv2 = _load_cv2()
    logger.info("Opening viewer for camera roles: %s", ", ".join(camera_roles))
    selected_cameras = [
        camera for camera in task_cfg.cameras if camera.role in set(camera_roles)
    ]
    subscribers = [
        (camera.role, RedisFrameSubscriber(task_cfg.redis, camera))
        for camera in selected_cameras
    ]
    cv2.namedWindow(_VIEWER_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if _viewer_window_closed(cv2):
                logger.info("Viewer window closed by the window manager")
                break

            panels = []
            for role, subscriber in subscribers:
                frame = subscriber.get_frame()
                if frame is None or frame.color is None:
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    panels.append(_overlay_status(blank, [role, "no frame available"]))
                    continue

                rgb = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
                lines = [
                    f"role: {role}",
                    f"frame: {frame.info.get('frame_id')}",
                    f"age_sec: {_safe_frame_age_sec(frame.info)}",
                    f"depth: {'yes' if frame.depth is not None else 'no'}",
                ]
                rgb = _overlay_status(rgb, lines)
                depth_vis = _normalize_depth(frame.depth)
                if depth_vis is not None:
                    panels.append(np.hstack([rgb, depth_vis]))
                else:
                    panels.append(rgb)

            canvas = _stack_panels_vertically(panels)
            display_canvas = _fit_canvas_for_display(canvas)
            cv2.imshow(_VIEWER_WINDOW_NAME, display_canvas)
            key = cv2.waitKey(1) & 0xFF
            if _viewer_window_closed(cv2):
                logger.info("Viewer window closed by the window manager")
                break
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        logger.info("Viewer interrupted from keyboard")
    finally:
        cv2.destroyAllWindows()
        logger.info("Viewer closed")
