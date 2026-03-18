#!/usr/bin/env python3
"""Estimate a static camera extrinsic from checkerboard or ChArUco images.

This script solves a standard PnP problem for one fixed camera:

1. Detect board corners in one or more images.
2. Estimate `camera_T_board` for each image.
3. Combine with a known `world_T_board` (or `robot_base_T_board`) transform.
4. Output the corresponding `world_T_camera` transform.

Conventions:
- `A_T_B` means "transform from frame B into frame A"
- 4x4 matrices are stored row-major
- translation units are meters
- rotation entries are unitless
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The `opencv-python` package is required for static camera calibration."
        ) from exc
    return cv2


def _load_realsense_report(path: Path, camera_role: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for camera in payload.get("cameras", []):
        if camera.get("role") == camera_role:
            return camera
    raise KeyError(f"Camera role `{camera_role}` not found in calibration report {path}")


def _intrinsics_for_image(camera_report: dict, image_width: int, image_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Choose native vs resized intrinsics that match the input image size."""

    rgb_stream = camera_report["rgb_stream"]
    for intrinsics_key in ("native_intrinsics", "resized_intrinsics"):
        intrinsics = rgb_stream[intrinsics_key]
        if int(intrinsics["width_px"]) == image_width and int(intrinsics["height_px"]) == image_height:
            camera_matrix = np.array(
                [
                    [float(intrinsics["fx_px"]), 0.0, float(intrinsics["ppx_px"])],
                    [0.0, float(intrinsics["fy_px"]), float(intrinsics["ppy_px"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            distortion = np.array(intrinsics["distortion_coeffs"], dtype=np.float64)
            return camera_matrix, distortion
    raise ValueError(
        f"No intrinsics in the report match image size {image_width}x{image_height}."
    )


def _object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Create checkerboard corner coordinates in the board frame."""

    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid[:, :2] *= float(square_size_m)
    return grid


def _aruco_dictionary(dictionary_name: str):
    cv2 = _load_cv2()
    for candidate in (dictionary_name, dictionary_name.upper()):
        if hasattr(cv2.aruco, candidate):
            return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, candidate))
    raise ValueError(f"Unsupported OpenCV ArUco/AprilTag dictionary `{dictionary_name}`.")


def _charuco_board(
    squares_x: int,
    squares_y: int,
    square_size_m: float,
    marker_size_m: float,
    dictionary_name: str,
):
    cv2 = _load_cv2()
    dictionary = _aruco_dictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_size_m),
        float(marker_size_m),
        dictionary,
    )
    return board, dictionary


def _matrix_from_row_major(values: Iterable[float]) -> np.ndarray:
    matrix = np.array(list(values), dtype=np.float64)
    if matrix.size != 16:
        raise ValueError("Expected exactly 16 values for a 4x4 row-major transform.")
    return matrix.reshape(4, 4)


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _load_cv2()
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def _average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    """Project the arithmetic mean back onto SO(3)."""

    mean_rotation = np.mean(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _average_transforms(transforms: List[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("At least one transform is required for averaging.")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _average_rotations([transform[:3, :3] for transform in transforms])
    result[:3, 3] = np.mean(np.stack([transform[:3, 3] for transform in transforms]), axis=0)
    return result


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3] @ b[:3, :3].T
    trace = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _solve_image_pose(
    image_path: Path,
    object_points: np.ndarray,
    checkerboard_cols: int,
    checkerboard_rows: int,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Dict[str, object]:
    cv2 = _load_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read calibration image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        (checkerboard_cols, checkerboard_rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise RuntimeError(f"Checkerboard was not detected in image: {image_path}")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError(f"solvePnP failed for image: {image_path}")

    reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    reprojection_error_px = float(
        np.mean(np.linalg.norm(reprojected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1))
    )
    return {
        "image": str(image_path),
        "camera_T_board": _transform_from_rvec_tvec(rvec, tvec),
        "reprojection_error_px": reprojection_error_px,
        "image_width_px": int(image.shape[1]),
        "image_height_px": int(image.shape[0]),
    }


def _solve_charuco_image_pose(
    image_path: Path,
    board,
    dictionary,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Dict[str, object]:
    cv2 = _load_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read calibration image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError(f"No ChArUco markers were detected in image: {image_path}")

    charuco_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
    )
    if charuco_ids is None or int(charuco_count) < 4:
        raise RuntimeError(
            f"Too few ChArUco corners were detected in image: {image_path}"
        )

    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        rvec=None,
        tvec=None,
    )
    if not success:
        raise RuntimeError(f"estimatePoseCharucoBoard failed for image: {image_path}")

    chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    object_points = chessboard_corners[charuco_ids.reshape(-1).astype(int)]
    reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    reprojection_error_px = float(
        np.mean(
            np.linalg.norm(
                reprojected.reshape(-1, 2) - charuco_corners.reshape(-1, 2), axis=1
            )
        )
    )
    return {
        "image": str(image_path),
        "camera_T_board": _transform_from_rvec_tvec(rvec, tvec),
        "reprojection_error_px": reprojection_error_px,
        "image_width_px": int(image.shape[1]),
        "image_height_px": int(image.shape[0]),
        "num_detected_corners": int(charuco_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-json", required=True, help="RealSense report JSON from export_realsense_calibration.py")
    parser.add_argument("--camera-role", required=True, help="Camera role, e.g. agentview")
    parser.add_argument("--images-dir", required=True, help="Directory containing checkerboard PNG/JPG images")
    parser.add_argument("--board-type", choices=["checkerboard", "charuco"], default="checkerboard")
    parser.add_argument("--checkerboard-cols", type=int, help="Number of checkerboard inner corners along board width")
    parser.add_argument("--checkerboard-rows", type=int, help="Number of checkerboard inner corners along board height")
    parser.add_argument("--square-size-m", type=float, required=True, help="Checkerboard / ChArUco square size in meters")
    parser.add_argument("--charuco-squares-x", type=int, help="Number of ChArUco squares along board width")
    parser.add_argument("--charuco-squares-y", type=int, help="Number of ChArUco squares along board height")
    parser.add_argument("--marker-size-m", type=float, help="ChArUco marker side length in meters")
    parser.add_argument("--aruco-dictionary", default="DICT_5X5_100", help="OpenCV ArUco / AprilTag dictionary name for ChArUco boards")
    parser.add_argument(
        "--reference-t-board",
        nargs=16,
        required=True,
        type=float,
        metavar=("r00", "r01", "r02", "tx", "r10", "r11", "r12", "ty", "r20", "r21", "r22", "tz", "m30", "m31", "m32", "m33"),
        help="Known 4x4 row-major transform such as world_T_board or robot_base_T_board.",
    )
    parser.add_argument("--reference-frame", default="world", help="Reference frame name, e.g. world or robot_base")
    parser.add_argument("--target-frame", default=None, help="Camera target frame name; defaults to <role>_camera")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    image_paths = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No PNG/JPG images found under {images_dir}")

    camera_report = _load_realsense_report(Path(args.calibration_json), args.camera_role)
    cv2 = _load_cv2()
    first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(f"Failed to read calibration image: {image_paths[0]}")
    camera_matrix, distortion = _intrinsics_for_image(
        camera_report,
        image_width=int(first_image.shape[1]),
        image_height=int(first_image.shape[0]),
    )

    object_points = None
    charuco_board = None
    charuco_dictionary = None
    if args.board_type == "checkerboard":
        if args.checkerboard_cols is None or args.checkerboard_rows is None:
            raise ValueError(
                "`--checkerboard-cols` and `--checkerboard-rows` are required for checkerboard calibration."
            )
        object_points = _object_points(
            cols=args.checkerboard_cols,
            rows=args.checkerboard_rows,
            square_size_m=args.square_size_m,
        )
    else:
        if (
            args.charuco_squares_x is None
            or args.charuco_squares_y is None
            or args.marker_size_m is None
        ):
            raise ValueError(
                "`--charuco-squares-x`, `--charuco-squares-y`, and `--marker-size-m` are required for ChArUco calibration."
            )
        charuco_board, charuco_dictionary = _charuco_board(
            squares_x=args.charuco_squares_x,
            squares_y=args.charuco_squares_y,
            square_size_m=args.square_size_m,
            marker_size_m=args.marker_size_m,
            dictionary_name=args.aruco_dictionary,
        )
    reference_T_board = _matrix_from_row_major(args.reference_t_board)

    per_image_results = []
    reference_T_camera_candidates = []
    for image_path in image_paths:
        if args.board_type == "checkerboard":
            image_result = _solve_image_pose(
                image_path=image_path,
                object_points=object_points,
                checkerboard_cols=args.checkerboard_cols,
                checkerboard_rows=args.checkerboard_rows,
                camera_matrix=camera_matrix,
                distortion=distortion,
            )
        else:
            image_result = _solve_charuco_image_pose(
                image_path=image_path,
                board=charuco_board,
                dictionary=charuco_dictionary,
                camera_matrix=camera_matrix,
                distortion=distortion,
            )
        camera_T_board = image_result["camera_T_board"]
        reference_T_camera = reference_T_board @ _invert_transform(camera_T_board)
        image_result["reference_T_camera"] = reference_T_camera.reshape(-1).tolist()
        per_image_results.append(image_result)
        reference_T_camera_candidates.append(reference_T_camera)

    reference_T_camera = _average_transforms(reference_T_camera_candidates)
    translation_errors_m = [
        float(np.linalg.norm(candidate[:3, 3] - reference_T_camera[:3, 3]))
        for candidate in reference_T_camera_candidates
    ]
    rotation_errors_deg = [
        _rotation_error_deg(candidate, reference_T_camera)
        for candidate in reference_T_camera_candidates
    ]

    target_frame = args.target_frame or f"{args.camera_role}_camera"
    payload = {
        "camera_role": args.camera_role,
        "reference_frame": args.reference_frame,
        "target_frame": target_frame,
        "board": (
            {
                "type": "checkerboard",
                "inner_corners_cols": args.checkerboard_cols,
                "inner_corners_rows": args.checkerboard_rows,
                "square_size_m": args.square_size_m,
            }
            if args.board_type == "checkerboard"
            else {
                "type": "charuco",
                "squares_x": args.charuco_squares_x,
                "squares_y": args.charuco_squares_y,
                "square_size_m": args.square_size_m,
                "marker_size_m": args.marker_size_m,
                "aruco_dictionary": args.aruco_dictionary,
            }
        ),
        "reference_T_board_row_major": reference_T_board.reshape(-1).tolist(),
        "reference_T_camera_row_major": reference_T_camera.reshape(-1).tolist(),
        "transform_name": f"{args.reference_frame}_T_{target_frame}",
        "per_image": [
            {
                "image": result["image"],
                "reprojection_error_px": result["reprojection_error_px"],
                "reference_T_camera_row_major": result["reference_T_camera"],
            }
            for result in per_image_results
        ],
        "summary": {
            "num_images": len(per_image_results),
            "mean_reprojection_error_px": float(
                np.mean([result["reprojection_error_px"] for result in per_image_results])
            ),
            "max_reprojection_error_px": float(
                np.max([result["reprojection_error_px"] for result in per_image_results])
            ),
            "mean_translation_deviation_m": float(np.mean(translation_errors_m)),
            "max_translation_deviation_m": float(np.max(translation_errors_m)),
            "mean_rotation_deviation_deg": float(np.mean(rotation_errors_deg)),
            "max_rotation_deviation_deg": float(np.max(rotation_errors_deg)),
        },
        "task_yaml_snippet": {
            "extrinsics": [
                {
                    "name": f"{args.reference_frame}_T_{target_frame}",
                    "reference_frame": args.reference_frame,
                    "target_frame": target_frame,
                    "transform": [float(value) for value in reference_T_camera.reshape(-1)],
                }
            ]
        },
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else images_dir / f"{args.camera_role}_static_extrinsic.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote static camera extrinsic estimate: {output_path}")
    print("")
    print("Task YAML snippet:")
    print(json.dumps(payload["task_yaml_snippet"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
