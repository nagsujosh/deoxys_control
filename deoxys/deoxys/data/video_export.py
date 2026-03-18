"""Export quick-look demo videos from `demo.hdf5` datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .logging_utils import get_data_logger
from .viewer import _load_cv2, _normalize_depth, _overlay_status, _stack_panels_vertically

logger = get_data_logger("video")


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("The `h5py` package is required to export demo videos.") from exc
    return h5py


def _demo_indices(data_group, demo_indices: Optional[Iterable[int]]) -> List[int]:
    if demo_indices is not None:
        requested = [int(index) for index in demo_indices]
        available = {
            int(name.split("_", 1)[1])
            for name in data_group.keys()
            if name.startswith("demo_") and name.split("_", 1)[1].isdigit()
        }
        missing = [index for index in requested if index not in available]
        if missing:
            raise KeyError(
                f"Requested demo indices are missing from the HDF5 file: {missing}. "
                f"Available demos: {sorted(available)}"
            )
        return requested
    indices = []
    for name in data_group.keys():
        if name.startswith("demo_") and name.split("_", 1)[1].isdigit():
            indices.append(int(name.split("_", 1)[1]))
    return sorted(indices)


def _video_canvas(demo_group, frame_idx: int, include_depth: bool) -> np.ndarray:
    cv2 = _load_cv2()
    obs = demo_group["obs"]
    camera_meta = demo_group.get("meta", {}).get("camera", {})
    panels = []
    for name, label in (
        ("agentview_rgb", "agentview"),
        ("eye_in_hand_rgb", "wrist"),
    ):
        if name not in obs:
            continue
        rgb = cv2.cvtColor(obs[name][frame_idx], cv2.COLOR_RGB2BGR)
        lines = [f"camera: {label}", f"frame: {frame_idx}"]
        if include_depth:
            depth_name = "agentview_depth" if name == "agentview_rgb" else "eye_in_hand_depth"
            depth_valid_name = f"{label}_depth_valid"
            depth_is_valid = True
            if depth_valid_name in camera_meta:
                depth_is_valid = bool(camera_meta[depth_valid_name][frame_idx].reshape(-1)[0])
            if depth_name in obs and depth_is_valid:
                depth_vis = _normalize_depth(obs[depth_name][frame_idx])
                if depth_vis is not None:
                    rgb = np.hstack([rgb, depth_vis])
                    lines.append("depth: included")
                else:
                    lines.append("depth: unavailable")
            elif depth_name in obs:
                lines.append("depth: placeholder")
            else:
                lines.append("depth: unavailable")
        else:
            lines.append("depth: disabled")
        rgb = _overlay_status(rgb, lines)
        panels.append(rgb)
    if not panels:
        raise ValueError("No supported RGB camera streams found in the requested demo.")
    return _stack_panels_vertically(panels)


def export_hdf5_demo_videos(
    hdf5_path: str | Path,
    output_dir: str | Path | None = None,
    demo_indices: Optional[Iterable[int]] = None,
    fps: float = 20.0,
    include_depth: bool = False,
) -> List[Path]:
    """Export one MP4 per demo group from `demo.hdf5`."""

    cv2 = _load_cv2()
    h5py = _load_h5py()

    hdf5_path = Path(hdf5_path).expanduser().resolve()
    if output_dir is None:
        output_dir = hdf5_path.parent / "demo_videos"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: List[Path] = []
    with h5py.File(hdf5_path, "r") as handle:
        data_group = handle["data"]
        for demo_index in _demo_indices(data_group, demo_indices):
            demo_group = data_group[f"demo_{demo_index}"]
            num_samples = int(demo_group.attrs["num_samples"])
            if num_samples <= 0:
                logger.warning(
                    "Skipping demo_%s in %s because it contains no frames",
                    demo_index,
                    hdf5_path,
                )
                continue
            if include_depth:
                available_depth_streams = [
                    name for name in ("agentview_depth", "eye_in_hand_depth") if name in demo_group["obs"]
                ]
                if available_depth_streams:
                    logger.info(
                        "Exporting demo_%s from %s with depth overlays for %s",
                        demo_index,
                        hdf5_path,
                        ", ".join(available_depth_streams),
                    )
                else:
                    logger.info(
                        "Depth was requested for demo_%s in %s, but no depth streams are present; "
                        "exporting RGB only",
                        demo_index,
                        hdf5_path,
                    )
            first_frame = _video_canvas(demo_group, 0, include_depth=include_depth)
            height, width = first_frame.shape[:2]
            output_path = output_dir / f"{hdf5_path.stem}_demo_{demo_index:03d}.mp4"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path}")

            try:
                for frame_idx in range(num_samples):
                    writer.write(_video_canvas(demo_group, frame_idx, include_depth=include_depth))
            finally:
                writer.release()

            exported_paths.append(output_path)
            logger.info("Exported demo video %s", output_path)
    return exported_paths
