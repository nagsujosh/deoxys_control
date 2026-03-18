#!/usr/bin/env python3
"""Estimate `ee_T_camera` from wrist-camera checkerboard or ChArUco samples.

This script uses OpenCV hand-eye calibration with samples containing:
- a wrist-camera image
- the corresponding measured robot pose `base_T_ee`

Workflow:
1. Detect the calibration board in each image.
2. Solve `camera_T_target` with OpenCV `solvePnP`.
3. Run `cv2.calibrateHandEye(...)`.
4. Return `ee_T_camera` (equivalently `gripper_T_camera`) as a 4x4 row-major transform.

Conventions:
- `A_T_B` means "transform from frame B into frame A"
- `base_T_ee` is the measured robot pose from Deoxys / Franka
- output `ee_T_camera` translation is in meters
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
            "The `opencv-python` package is required for hand-eye calibration."
        ) from exc
    return cv2


def _load_realsense_report(path: Path, camera_role: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for camera in payload.get("cameras", []):
        if camera.get("role") == camera_role:
            return camera
    raise KeyError(f"Camera role `{camera_role}` not found in calibration report {path}")


def _intrinsics_for_image(camera_report: dict, image_width: int, image_height: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _load_cv2()
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3] @ b[:3, :3].T
    trace = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _solve_camera_T_target(
    image_path: Path,
    object_points: np.ndarray,
    checkerboard_cols: int,
    checkerboard_rows: int,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Tuple[np.ndarray, float]:
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
    return _transform_from_rvec_tvec(rvec, tvec), reprojection_error_px


def _solve_charuco_camera_T_target(
    image_path: Path,
    board,
    dictionary,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Tuple[np.ndarray, float]:
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
    return _transform_from_rvec_tvec(rvec, tvec), reprojection_error_px


def _method_flag(name: str) -> int:
    cv2 = _load_cv2()
    mapping = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported hand-eye method `{name}`.") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-json", required=True, help="RealSense report JSON from export_realsense_calibration.py")
    parser.add_argument("--camera-role", default="wrist", help="Camera role to calibrate, usually wrist")
    parser.add_argument(
        "--samples-json",
        required=True,
        help="Sample manifest JSON from capture_calibration_samples.py with `base_T_ee` entries.",
    )
    parser.add_argument("--board-type", choices=["checkerboard", "charuco"], default="checkerboard")
    parser.add_argument("--checkerboard-cols", type=int, help="Number of checkerboard inner corners along board width")
    parser.add_argument("--checkerboard-rows", type=int, help="Number of checkerboard inner corners along board height")
    parser.add_argument("--square-size-m", required=True, type=float, help="Checkerboard / ChArUco square size in meters")
    parser.add_argument("--charuco-squares-x", type=int, help="Number of ChArUco squares along board width")
    parser.add_argument("--charuco-squares-y", type=int, help="Number of ChArUco squares along board height")
    parser.add_argument("--marker-size-m", type=float, help="ChArUco marker side length in meters")
    parser.add_argument("--aruco-dictionary", default="DICT_5X5_100", help="OpenCV ArUco / AprilTag dictionary name for ChArUco boards")
    parser.add_argument(
        "--method",
        default="tsai",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        help="OpenCV hand-eye method",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    samples_payload = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))
    samples = samples_payload.get("samples", [])
    if len(samples) < 3:
        raise ValueError("Hand-eye calibration requires at least 3 samples; 8+ is much better.")

    first_image = _load_cv2().imread(str(samples[0]["image"]), _load_cv2().IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(f"Failed to read calibration image: {samples[0]['image']}")

    camera_report = _load_realsense_report(Path(args.calibration_json), args.camera_role)
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
                "`--checkerboard-cols` and `--checkerboard-rows` are required for checkerboard hand-eye calibration."
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
                "`--charuco-squares-x`, `--charuco-squares-y`, and `--marker-size-m` are required for ChArUco hand-eye calibration."
            )
        charuco_board, charuco_dictionary = _charuco_board(
            squares_x=args.charuco_squares_x,
            squares_y=args.charuco_squares_y,
            square_size_m=args.square_size_m,
            marker_size_m=args.marker_size_m,
            dictionary_name=args.aruco_dictionary,
        )

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    per_sample = []

    for sample in samples:
        if "base_T_ee" not in sample:
            raise KeyError(
                "Every hand-eye sample must include `base_T_ee`. "
                "Capture them with `capture_calibration_samples.py --with-robot-pose`."
            )
        base_T_ee = _matrix_from_row_major(sample["base_T_ee"])
        if args.board_type == "checkerboard":
            camera_T_target, reprojection_error_px = _solve_camera_T_target(
                image_path=Path(sample["image"]),
                object_points=object_points,
                checkerboard_cols=args.checkerboard_cols,
                checkerboard_rows=args.checkerboard_rows,
                camera_matrix=camera_matrix,
                distortion=distortion,
            )
        else:
            camera_T_target, reprojection_error_px = _solve_charuco_camera_T_target(
                image_path=Path(sample["image"]),
                board=charuco_board,
                dictionary=charuco_dictionary,
                camera_matrix=camera_matrix,
                distortion=distortion,
            )
        R_gripper2base.append(base_T_ee[:3, :3])
        t_gripper2base.append(base_T_ee[:3, 3])
        R_target2cam.append(camera_T_target[:3, :3])
        t_target2cam.append(camera_T_target[:3, 3])
        per_sample.append(
            {
                "image": sample["image"],
                "base_T_ee_row_major": base_T_ee.reshape(-1).tolist(),
                "camera_T_target_row_major": camera_T_target.reshape(-1).tolist(),
                "reprojection_error_px": reprojection_error_px,
            }
        )

    cv2 = _load_cv2()
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=_method_flag(args.method),
    )

    ee_T_camera = np.eye(4, dtype=np.float64)
    ee_T_camera[:3, :3] = np.asarray(R_cam2gripper, dtype=np.float64)
    ee_T_camera[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)

    # For a fixed target, each sample should imply a similar base_T_target:
    # base_T_target = base_T_ee @ ee_T_camera @ camera_T_target
    base_T_target_estimates = [
        _matrix_from_row_major(entry["base_T_ee_row_major"]) @ ee_T_camera @ _matrix_from_row_major(entry["camera_T_target_row_major"])
        for entry in per_sample
    ]
    mean_base_T_target = np.eye(4, dtype=np.float64)
    mean_base_T_target[:3, 3] = np.mean(
        np.stack([estimate[:3, 3] for estimate in base_T_target_estimates]), axis=0
    )
    mean_base_T_target[:3, :3] = np.mean(
        np.stack([estimate[:3, :3] for estimate in base_T_target_estimates]), axis=0
    )
    u, _, vt = np.linalg.svd(mean_base_T_target[:3, :3])
    mean_base_T_target[:3, :3] = u @ vt

    translation_errors_m = [
        float(np.linalg.norm(estimate[:3, 3] - mean_base_T_target[:3, 3]))
        for estimate in base_T_target_estimates
    ]
    rotation_errors_deg = [
        _rotation_error_deg(estimate, mean_base_T_target)
        for estimate in base_T_target_estimates
    ]

    payload = {
        "camera_role": args.camera_role,
        "handeye_method": args.method,
        "ee_T_camera_row_major": ee_T_camera.reshape(-1).tolist(),
        "transform_name": f"ee_T_{args.camera_role}_camera",
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
        "summary": {
            "num_samples": len(per_sample),
            "mean_reprojection_error_px": float(np.mean([entry["reprojection_error_px"] for entry in per_sample])),
            "max_reprojection_error_px": float(np.max([entry["reprojection_error_px"] for entry in per_sample])),
            "mean_implied_target_translation_std_m": float(np.mean(translation_errors_m)),
            "max_implied_target_translation_std_m": float(np.max(translation_errors_m)),
            "mean_implied_target_rotation_std_deg": float(np.mean(rotation_errors_deg)),
            "max_implied_target_rotation_std_deg": float(np.max(rotation_errors_deg)),
        },
        "per_sample": per_sample,
        "task_yaml_snippet": {
            "extrinsics": [
                {
                    "name": f"ee_T_{args.camera_role}_camera",
                    "reference_frame": "ee",
                    "target_frame": f"{args.camera_role}_camera",
                    "transform": [float(value) for value in ee_T_camera.reshape(-1)],
                }
            ]
        },
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else Path(args.samples_json).expanduser().resolve().parent / f"{args.camera_role}_handeye_extrinsic.json"
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote hand-eye calibration result: {output_path}")
    print("")
    print("Task YAML snippet:")
    print(json.dumps(payload["task_yaml_snippet"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
