# Camera Calibration Notes

This directory is the place to look when you want to gather or fill camera
geometry metadata for the teleoperation pipeline.

## Scripts in this folder

- `export_realsense_calibration.py`
  Exports the RealSense-native intrinsics, depth scale, and sensor-to-sensor
  extrinsics that the device already knows.
- `capture_calibration_samples.py`
  Captures checkerboard images from one Redis-published camera stream and can
  optionally save matching `base_T_ee` robot poses for hand-eye calibration.
- `estimate_static_camera_extrinsic.py`
  Solves a static-camera extrinsic such as `world_T_agentview` from checkerboard
  or ChArUco images plus a known `world_T_board`.
- `estimate_handeye_extrinsic.py`
  Solves a wrist-camera extrinsic such as `ee_T_wrist_camera` from checkerboard
  or ChArUco images plus recorded `base_T_ee` robot poses.

## What you can get automatically

The script in this folder can export the calibration that the RealSense device
already knows:

- RGB intrinsics
- depth intrinsics
- resized intrinsics that match the dataset image size
- depth-to-color and color-to-depth sensor extrinsics
- depth scale

Run it like this:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/export_realsense_calibration.py --task fr3_dual_realsense
```

Outputs go to:

- `tools/calibration/output/<task>_realsense_calibration_<timestamp>.json`
- `tools/calibration/output/<task>_extrinsics_template.yml`

## Capture calibration samples

Before running the estimator scripts, you usually need a small set of
calibration-board snapshots.

### Static / external camera capture

Make sure `deoxys.data up --task <task>` is running, then capture snapshots:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/capture_calibration_samples.py \
  --task fr3_dual_realsense \
  --camera-role agentview \
  --output-dir tools/calibration/output/agentview_static_samples
```

Controls:

- `SPACE`: save one PNG image
- `Q` or `ESC`: quit

This writes:

- PNG images such as `agentview_0000.png`
- `samples.json`

### Wrist / hand camera capture

This mode also records the latest measured `base_T_ee` pose from Deoxys:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/capture_calibration_samples.py \
  --task fr3_dual_realsense \
  --camera-role wrist \
  --output-dir tools/calibration/output/wrist_handeye_samples \
  --with-robot-pose
```

For good hand-eye calibration:

- move the wrist through many distinct poses
- keep the calibration board rigid in the scene
- capture at least 8-12 varied poses
- avoid taking all samples from nearly the same orientation

## What you still need to calibrate yourself

The RealSense camera does **not** know where it is relative to your:

- world frame
- robot base frame
- end-effector frame

Those transforms must come from your own setup calibration.

Typical examples:

- `world_T_agentview`
- `robot_base_T_agentview`
- `ee_T_wrist_camera`

## How to get the missing matrix

### External / static camera

Goal:
- find `world_T_camera` or `robot_base_T_camera`

Typical method:
1. Place a checkerboard or ChArUco / AprilTag-dictionary board at a known pose in the robot/world frame.
2. Capture one or more images from the static camera.
3. Detect the target corners or tags.
4. Solve the camera pose with `solvePnP`, an AprilTag pose pipeline, Kalibr, or a comparable calibration tool.
5. Convert the result into a 4x4 homogeneous transform in row-major order.

Checkerboard version:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/estimate_static_camera_extrinsic.py \
  --calibration-json tools/calibration/output/fr3_dual_realsense_realsense_calibration_YYYYMMDD_HHMMSS.json \
  --camera-role agentview \
  --images-dir tools/calibration/output/agentview_static_samples \
  --board-type checkerboard \
  --checkerboard-cols 9 \
  --checkerboard-rows 6 \
  --square-size-m 0.025 \
  --reference-frame world \
  --reference-t-board 1 0 0 0.50 0 1 0 0.10 0 0 1 0.02 0 0 0 1
```

ChArUco / AprilTag-dictionary board version:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/estimate_static_camera_extrinsic.py \
  --calibration-json tools/calibration/output/fr3_dual_realsense_realsense_calibration_YYYYMMDD_HHMMSS.json \
  --camera-role agentview \
  --images-dir tools/calibration/output/agentview_static_samples \
  --board-type charuco \
  --charuco-squares-x 7 \
  --charuco-squares-y 5 \
  --square-size-m 0.040 \
  --marker-size-m 0.030 \
  --aruco-dictionary DICT_APRILTAG_36h11 \
  --reference-frame world \
  --reference-t-board 1 0 0 0.50 0 1 0 0.10 0 0 1 0.02 0 0 0 1
```

Notes:

- `reference_t_board` is the known `world_T_board` or `robot_base_T_board`
- the output JSON includes a task-YAML snippet ready to paste
- the script chooses native vs resized intrinsics based on the input image size

### Wrist / hand camera

Goal:
- find `ee_T_wrist_camera`

Typical method:
1. Mount a calibration target rigidly in the scene.
2. Move the robot through multiple distinct wrist poses.
3. For each pose, record:
   - the robot end-effector pose from Deoxys / Franka
   - the observed target pose in the wrist camera
4. Solve the hand-eye problem `AX = XB`.
5. Convert the result into a 4x4 homogeneous transform in row-major order.

Checkerboard + OpenCV hand-eye version:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/estimate_handeye_extrinsic.py \
  --calibration-json tools/calibration/output/fr3_dual_realsense_realsense_calibration_YYYYMMDD_HHMMSS.json \
  --camera-role wrist \
  --samples-json tools/calibration/output/wrist_handeye_samples/samples.json \
  --board-type checkerboard \
  --checkerboard-cols 9 \
  --checkerboard-rows 6 \
  --square-size-m 0.025 \
  --method tsai
```

ChArUco / AprilTag-dictionary board hand-eye version:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
python tools/calibration/estimate_handeye_extrinsic.py \
  --calibration-json tools/calibration/output/fr3_dual_realsense_realsense_calibration_YYYYMMDD_HHMMSS.json \
  --camera-role wrist \
  --samples-json tools/calibration/output/wrist_handeye_samples/samples.json \
  --board-type charuco \
  --charuco-squares-x 7 \
  --charuco-squares-y 5 \
  --square-size-m 0.040 \
  --marker-size-m 0.030 \
  --aruco-dictionary DICT_APRILTAG_36h11 \
  --method tsai
```

For tag-board calibration, the key parameters you need are:

- dictionary family, for example `DICT_APRILTAG_36h11`
- number of board squares along `x` and `y`
- square size in meters
- marker side length in meters

Input requirement:

- each sample in `samples.json` must contain `base_T_ee`
- `capture_calibration_samples.py --with-robot-pose` produces that format

Output:

- JSON result file with `ee_T_camera`
- task-YAML snippet ready to paste into the wrist camera `extrinsics:` list

Common tools:

- OpenCV hand-eye calibration
- MoveIt calibration
- `easy_handeye`
- Kalibr
- custom AprilTag-based pose estimation + hand-eye solver

## Where to paste the matrix afterward

Put the final setup extrinsics into:

- [fr3_dual_realsense.yml](/home/carl/deoxys_control/deoxys/config/data_tasks/fr3_dual_realsense.yml)

The camera section supports:

```yaml
extrinsics:
  - name: world_T_agentview
    reference_frame: world
    target_frame: agentview_camera
    transform: [ ... 16 numbers ... ]
```

Notes:

- `transform` must be a 4x4 homogeneous transform in row-major order
- translation units must be meters
- rotation entries are unitless
- do not use identity as a placeholder unless the identity is actually correct

## Why this matters

These calibration fields are now propagated into:

- raw run outputs as `testing_demo_camera_<id>_calibration.json`
- `manifest.json`
- HDF5 under `meta/camera/<role>/calibration`

That makes later geometry-aware exports such as Zarr much easier and less error-prone.
