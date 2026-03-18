# Deoxys Teleop Data Pipeline Guide

This guide covers the full desktop-side workflow for the new `deoxys.data`
pipeline:

1. Start Redis and both RealSense camera publishers
2. Verify live RGB-D streams
3. Reset the robot to a known preset
4. Use plain SpaceMouse teleoperation or collect teleoperation data
5. Validate raw `runN` folders before dataset build, with optional camera replay
6. Replay a stored raw run on the robot using its saved delta actions
7. Build `demo.hdf5` from raw `runN` folders
8. Export quick-look MP4 videos from `demo.hdf5`

This is the canonical workflow for local data collection on this machine.

## Environment

Activate the validated Conda environment:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
```

The editable install already exposes the CLI as:

```bash
deoxys.data --help
```

Helpful shared-lab discovery commands:

```bash
deoxys.data tasks
deoxys.data dates --task fr3_dual_realsense
deoxys.data runs --task fr3_dual_realsense --all-dates
deoxys.data task-create --task bell_pepper_pick --from-task fr3_dual_realsense
```

## Config Files

The new pipeline uses the existing Deoxys config root:

- Task configs: [deoxys/config/data_tasks](/home/carl/deoxys_control/deoxys/config/data_tasks)
- Reset presets: [deoxys/config/data_resets](/home/carl/deoxys_control/deoxys/config/data_resets)
- Robot/network config: [deoxys/config/charmander.yml](/home/carl/deoxys_control/deoxys/config/charmander.yml)
- Controller config: [deoxys/config/osc-pose-controller.yml](/home/carl/deoxys_control/deoxys/config/osc-pose-controller.yml)
- Reset controller config: [deoxys/config/joint-position-controller.yml](/home/carl/deoxys_control/deoxys/config/joint-position-controller.yml)

Default task:

- [fr3_dual_realsense.yml](/home/carl/deoxys_control/deoxys/config/data_tasks/fr3_dual_realsense.yml)

Default reset preset:

- [home_nominal.yml](/home/carl/deoxys_control/deoxys/config/data_resets/home_nominal.yml)

The default task config sets `default_reset_preset: home_nominal`, so
`teleop`, `collect`, and `replay` all start from the same nominal joint
configuration unless you override the preset explicitly.

Current RealSense serial mapping in the task config:

- `agentview` -> `999999999999`
- `wrist` -> `888888888888`

Default image settings in the validated task config:

- capture resolution: `640x480`
- saved RGB-D resolution: `640x480`
- color convention: RGB in memory and on disk; internal OpenCV encode/decode steps are converted back to RGB so the dataset is RGB-D, not BGR-D
- frame rate: `30 Hz` per camera

Current robot/control timing from
[charmander.yml](/home/carl/deoxys_control/deoxys/config/charmander.yml):

- state publisher: `100 Hz`
- policy / teleoperation control loop: `20 Hz`
- trajectory interpolation: `500 Hz`
- nominal saved dataset action/state rate: about `20 Hz`

If the physical camera mounting is reversed, swap the `serial_number` fields in
[fr3_dual_realsense.yml](/home/carl/deoxys_control/deoxys/config/data_tasks/fr3_dual_realsense.yml).

## Creating a New Task

To create a new collection task such as `bell_pepper_pick`, add a new task YAML
under [deoxys/config/data_tasks](/home/carl/deoxys_control/deoxys/config/data_tasks).

Example file:

- `deoxys/config/data_tasks/bell_pepper_pick.yml`

The CLI task name comes from the filename stem:

```bash
deoxys.data collect --task bell_pepper_pick
```

Inside that YAML, these two fields control where data is written:

- `name`: dataset/task name used in the save path
- `output_root`: base output directory

Example task header:

```yaml
name: bell_pepper_pick
output_root: data
interface_cfg: charmander.yml
controller_type: OSC_POSE
controller_cfg: osc-pose-controller.yml
default_reset_preset: home_nominal
```

With that configuration, the collector writes runs to:

```text
/home/carl/deoxys_control/data/bell_pepper_pick/YYYY-MM-DD/run1
/home/carl/deoxys_control/data/bell_pepper_pick/YYYY-MM-DD/run2
...
```

And the HDF5 builder writes:

```text
/home/carl/deoxys_control/data/bell_pepper_pick/YYYY-MM-DD/demo.hdf5
```

Important detail:

- `--task bell_pepper_pick` selects `bell_pepper_pick.yml`
- `name: bell_pepper_pick` controls the task folder name inside `output_root`
- if the YAML filename and `name:` differ, the saved dataset path uses `name:`
- the pipeline still uses the YAML filename stem for CLI lookup and managed child
  services, so those two identifiers no longer conflict in normal use

If you want the CLI to scaffold a new task YAML automatically from an existing
task template, use:

```bash
deoxys.data task-create --task bell_pepper_pick --from-task fr3_dual_realsense
```

## Scenario: Make a New Reset Preset for One Task

This is the most common case in a shared lab: you want a safer or lower reset
pose for one task, but you do not want to change every other task.

Recommended workflow:

1. Create a new task YAML from an existing task template.
2. Create a new reset preset YAML with the new joint targets.
3. Test that reset directly with `deoxys.data reset`.
4. Set the new task’s `default_reset_preset` to that new preset.
5. Use `teleop`, `collect`, or `replay` on the new task.

Example:

```bash
deoxys.data task-create --task bell_pepper_pick --from-task fr3_dual_realsense
```

Then create a new reset preset under
[deoxys/config/data_resets](/home/carl/deoxys_control/deoxys/config/data_resets),
for example `home_bell_pepper_low.yml`:

```yaml
name: home_bell_pepper_low
controller_type: JOINT_POSITION
controller_cfg: joint-position-controller.yml
joint_positions: [0.09, -0.98, 0.03, -2.69, 0.05, 1.86, 0.87]
tolerance_rad: 0.001
timeout_sec: 30.0
allow_jitter: false
jitter_std_rad: 0.0
jitter_clip_rad: 0.0
```

Test it:

```bash
deoxys.data reset --task bell_pepper_pick --preset home_bell_pepper_low
```

If the pose is correct, update the task YAML
`deoxys/config/data_tasks/bell_pepper_pick.yml`:

```yaml
default_reset_preset: home_bell_pepper_low
```

After that, these commands automatically use the new reset:

```bash
deoxys.data teleop --task bell_pepper_pick
deoxys.data collect --task bell_pepper_pick
deoxys.data replay --task bell_pepper_pick --date 2026-03-18 --run run1
```

This keeps reset behavior task-specific instead of changing the global default
for every other experiment.

## What Each Command Does

### `preflight`

Checks whether Redis is reachable and whether each configured camera topic is
fresh enough according to the task config timeout.

```bash
deoxys.data preflight --task fr3_dual_realsense
```

Expected result:

- exit code `0` when Redis is reachable and both camera streams are fresh
- exit code `1` when Redis is down or any required stream is stale/missing

### `up`

Starts managed Redis plus one RealSense publisher process per configured camera.
This command keeps running and prints a health report once per second.

```bash
deoxys.data up --task fr3_dual_realsense
```

Use `Ctrl+C` to stop the managed processes cleanly.

### `view`

Displays one or both live Redis camera streams in OpenCV windows with overlays
for role, frame id, timestamp freshness, and depth availability.

```bash
deoxys.data view --task fr3_dual_realsense --camera-role all
```

Single-camera examples:

```bash
deoxys.data view --task fr3_dual_realsense --camera-role agentview
deoxys.data view --task fr3_dual_realsense --camera-role wrist
```

### `reset`

Moves the arm to the named joint reset preset using the joint-position
controller. Joint targets in reset presets are stored in radians.

```bash
deoxys.data reset --task fr3_dual_realsense --preset home_nominal
```

### `collect`

Starts one teleoperation collection run. The collector:

1. runs the task's default reset preset, or an explicit `--preset` override
2. explicitly opens the gripper and verifies it reached a near-fully-open width
3. warms up the interfaces
4. waits for first real motion
5. records accepted samples only
6. writes one `runN` directory with raw arrays and `manifest.json`

```bash
deoxys.data collect --task fr3_dual_realsense
deoxys.data collect --task fr3_dual_realsense --no-reset
deoxys.data collect --task fr3_dual_realsense --inverse
```

Use the SpaceMouse right button to finish the current demo and save it.
Use `Ctrl+C` only when you want to discard the current run and remove the
partial `runN` directory instead of saving partial files or metadata.
The collector emits one short terminal bell when it transitions into active
recording after the first valid motion, and a two-beep confirmation when the
finish-and-save signal is accepted from the SpaceMouse right button.
When a run is discarded or cleaned up after failure, the CLI reports
`run_dir: null` instead of pointing at a deleted path.
Pass `--no-reset` when you want to skip the task's automatic reset and start
collecting immediately from the robot's current pose.
Pass `--inverse` when you want the 6 arm-motion channels inverted for that CLI
run only; gripper semantics stay normal, the task YAML remains unchanged, and
the default mapping stays normal.

### `teleop`

Runs plain SpaceMouse teleoperation without recording a dataset. This uses the
same task config, SpaceMouse ids, controller type, interface config, and action
transform settings as the collector, and it also runs the task's default reset
preset first when one is configured. Before teleoperation becomes active, it
explicitly opens the gripper and verifies it reached a near-fully-open width.
If the gripper cannot be verified open, teleoperation aborts cleanly instead of
starting anyway.

```bash
deoxys.data teleop --task fr3_dual_realsense
deoxys.data teleop --task fr3_dual_realsense --no-reset
deoxys.data teleop --task fr3_dual_realsense --inverse
```

Optional bounded run:

```bash
deoxys.data teleop --task fr3_dual_realsense --max-steps 500
```

Optional preset override:

```bash
deoxys.data teleop --task fr3_dual_realsense --preset home_nominal
```

Use the SpaceMouse right button or `Ctrl+C` to stop plain teleoperation cleanly.
Pass `--no-reset` when you want to skip the task's automatic reset first.
Pass `--inverse` when you want the 6 arm-motion channels inverted for that CLI
run only; gripper semantics stay normal, the task YAML remains unchanged, and
the default mapping stays normal.

### `validate`

Validates one raw run and prints a structured report covering stream lengths,
timing skew/age summaries, missing-frame counts, suspicious near-zero state
segments, and can optionally replay the raw camera streams.

```bash
deoxys.data validate --task fr3_dual_realsense --date 2026-03-15 --run run1
deoxys.data validate --task fr3_dual_realsense --date 2026-03-15 --run run1 --play --fps 10
```

If `--run` is omitted, the latest run under the selected date is used. If
`--date` is also omitted, the latest task date directory is used.

Optional depth playback:

```bash
deoxys.data validate --task fr3_dual_realsense --run run1 --play --include-depth
```

### `replay`

Replays one stored raw run on the robot using the canonical saved delta-action
stream from `testing_demo_delta_action.npz`.

```bash
deoxys.data replay --task fr3_dual_realsense --date 2026-03-15 --run run1
deoxys.data replay --task fr3_dual_realsense --date 2026-03-15 --run run1 --no-reset
```

Optional preset override:

```bash
deoxys.data replay --task fr3_dual_realsense --date 2026-03-15 --run run1 --preset home_nominal
```

If `--run` is omitted, the latest run under the selected date is used. If
`--date` is also omitted, the latest task date directory is used.
Pass `--no-reset` when you want replay to begin from the robot's current pose
instead of running the task's default reset preset first.

### `build`

Builds `demo.hdf5` from all valid `runN` folders under one task/date root.

```bash
deoxys.data build --task fr3_dual_realsense --date 2026-03-14 --overwrite
```

`--overwrite` is required when `demo.hdf5` already exists.

You can also aggregate several dates for the same task into one task-level
dataset:

```bash
deoxys.data build --task fr3_dual_realsense --all-dates --overwrite
deoxys.data build --task fr3_dual_realsense --dates 2026-03-17 2026-03-18 --overwrite
```

### `video`

Exports quick-look MP4 files from a built `demo.hdf5`.

```bash
deoxys.data video --hdf5 /home/carl/deoxys_control/data/fr3_dual_realsense/2026-03-15/demo.hdf5
```

The equivalent standalone script is:

```bash
python tools/data/export_hdf5_demo_video.py --hdf5 /path/to/demo.hdf5
```

Add `--include-depth` to append available depth visualizations next to RGB.
If depth is requested but a demo or camera stream does not actually contain a
depth dataset, the exporter falls back to RGB for that stream and logs that
depth was unavailable.

## Recommended Functional Workflow

Open separate terminals as needed.

### 1. Bring up the NUC-side interfaces

Run on the Intel NUC:

```bash
./franka_init.sh
```

### 2. Start the desktop pipeline services

Run on the desktop:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
deoxys.data up --task fr3_dual_realsense
```

Leave this terminal running.

### 3. Verify the streams

In a second terminal:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
deoxys.data preflight --task fr3_dual_realsense
deoxys.data view --task fr3_dual_realsense --camera-role all
```

Confirm both cameras are live and look correct before collecting data.

### 4. Optional manual reset check

In a third terminal:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
deoxys.data reset --task fr3_dual_realsense --preset home_nominal
```

This step is optional when you are about to run `teleop`, `collect`, or
`replay`, because those commands already use the task's `default_reset_preset`
by default. It is still useful if you want to manually verify the nominal joint
posture before starting a session.

### 5. Collect teleoperation data

Still on the desktop:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
deoxys.data collect --task fr3_dual_realsense
```

Operate the robot with the SpaceMouse.

The collector rejects invalid samples automatically when:

- robot state is missing
- required RGB is missing
- required depth is missing for a depth-required camera
- no real motion has happened yet

Quality-oriented collection defaults:

- the collector records `(obs_t, delta_action_t)` before sending the control command for that step
- `teleop`, `collect`, and `replay` all start from the same reset posture when the task config defines `default_reset_preset`
- near-zero robot state reads can reuse the previous valid state when configured
- interrupting collection with `Ctrl+C` discards the run and removes the partial output directory instead of saving partial data
- pressing the SpaceMouse right button finishes the current demo and saves it when valid samples were recorded
- collection failures return a structured `failure_reason` in CLI output
- failed runs are deleted by default unless the task config opts in to keeping them
- `collection.keep_failed_runs` applies to genuine collection failures, not user-canceled runs
- optional secondary cameras can be listed in `collection.optional_camera_roles`; the collector keeps alignment with placeholder frames and saves a validity mask instead of dropping samples
- depth-enabled cameras also write a separate depth-validity mask so zero-filled placeholder depth images are never mistaken for real depth frames
- if you want temporary arm-motion inversion from the CLI, use `--inverse` on `teleop` or `collect`
- if you need a persistent custom inversion, use `collection.action_multipliers` in the task YAML

### 6. Validate the raw run

Before building HDF5, validate the new run:

```bash
deoxys.data validate --task fr3_dual_realsense --date $(date +%F) --run run1
```

And optionally replay it:

```bash
deoxys.data validate --task fr3_dual_realsense --date $(date +%F) --run run1 --play
```

`deoxys.data` is the canonical workflow for teleoperation, collection, replay,
validation, and HDF5 build. Older scripts under `deoxys/examples/` are legacy
references and may keep their own manual save prompts or example-specific
behavior.

### 7. Replay the robot run if needed

If you want to physically replay a recorded demonstration on the robot:

```bash
deoxys.data replay --task fr3_dual_realsense --date $(date +%F) --run run1
```

This uses the saved delta-action stream directly.

### 8. Build the dataset

After recording runs:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
deoxys.data build --task fr3_dual_realsense --date $(date +%F) --overwrite
```

### 9. Export demo videos

```bash
deoxys.data video --hdf5 /home/carl/deoxys_control/data/fr3_dual_realsense/$(date +%F)/demo.hdf5
```

## Output Layout

The collector writes to:

```text
data/<task>/<YYYY-MM-DD>/runN/
```

Example:

```text
data/fr3_dual_realsense/2026-03-14/run1/
```

For a custom task such as `bell_pepper_pick` with `output_root: data`, the
full layout becomes:

```text
data/bell_pepper_pick/2026-03-15/run1/
data/bell_pepper_pick/2026-03-15/run2/
data/bell_pepper_pick/2026-03-15/demo.hdf5
```

Each run stores the lean raw tensors needed for RGB-D imitation learning plus calibration metadata:

- `testing_demo_delta_action.npz`
- `testing_demo_ee_state_10d.npz`
- `testing_demo_camera_0_color.npz`
- `testing_demo_camera_1_color.npz`
- `testing_demo_camera_0_valid.npz`
- `testing_demo_camera_1_valid.npz`
- `testing_demo_camera_0_calibration.json`
- `testing_demo_camera_1_calibration.json`
- `testing_demo_camera_0_depth.npz` and `testing_demo_camera_0_depth_valid.npz` when depth is enabled
- `testing_demo_camera_1_depth.npz` and `testing_demo_camera_1_depth_valid.npz` when depth is enabled
- `manifest.json`

Older runs collected before this cleanup may still contain extra debug/QA artifacts such as
tracking errors, native pose matrices, sync metadata, or robot metadata. The
builder and validator can still read those older runs, but newly collected runs
no longer save those larger side-channel files by default.

The HDF5 builder writes:

```text
data/<task>/<YYYY-MM-DD>/demo.hdf5
```

## Action Semantics

For the default `OSC_POSE` task, each saved action is a 7D delta-action vector:

```text
[dx, dy, dz, d_rot_0, d_rot_1, d_rot_2, gripper]
```

Meaning:

- `dx`, `dy`, `dz`: Cartesian translation command dimensions used by the
  Deoxys OSC controller. These are saved before the controller applies
  `action_scale.translation`.
- `d_rot_0`, `d_rot_1`, `d_rot_2`: the three controller-native rotational
  command channels after the SpaceMouse axis reorder/sign convention in
  [input_utils.py](/home/carl/deoxys_control/deoxys/deoxys/utils/input_utils.py).
  These are saved before the controller applies `action_scale.rotation`.
- `gripper`: scalar gripper command. Nonnegative values mean close/grasp;
  negative values mean open/release.

Frame and convention notes:

- The saved action lives in the controller command convention used by the
  selected Deoxys controller, not in raw HID device coordinates.
- For the default OSC controllers, this means the saved action is an
  end-effector/controller delta command after SpaceMouse mapping, not the raw
  SpaceMouse delta itself.
- The pipeline does not store Euler roll/pitch/yaw orientation state features.
  Orientation state features are stored as rotation matrices and 6D rotation
  representations.
- For `OSC_POSE`, the collector stores the exact 7D delta-action vector passed
  to `FrankaInterface.control(...)` before controller-side scaling.
- The rotational command part is still 3D because that is what the Deoxys
  controller interface accepts; there is no controller-native 6D action input.
- The default controller scaling comes from
  [osc-pose-controller.yml](/home/carl/deoxys_control/deoxys/config/osc-pose-controller.yml):
  translation scale `0.05`, rotation scale `1.0`.
- The collector writes one canonical `delta_actions` stream, and that stream is the
  delta action used for HDF5 export and robot replay.

## Learning-Facing State Layout

The pipeline now exposes the end-effector state in one main learning-facing form:

- `ee_state_10d`: `3 position + 6 rotation + 1 gripper width = 10D`

For imitation-learning models, `ee_state_10d` is the default combined state
feature when you want end-effector pose and gripper in one tensor.

## Dataset Semantics

The dataset includes:

- RGB observations for `agentview` and `wrist`
- depth observations when enabled
- one canonical delta-action stream stored as `delta_actions` in HDF5
- `ee_state_10d`: xyz position plus 6D rotation plus gripper width
- per-camera validity masks
- camera calibration metadata for downstream geometry-aware exports such as Zarr

Important pose note:

- Franka/Deoxys natively exposes pose matrices such as `O_T_EE`
- there is no native Franka-provided "6D rotation representation" field for learning
- the pipeline derives `ee_state_10d` from the measured pose matrix plus measured gripper width

Important unit conventions:

- joint position: radians
- joint velocity: radians/second
- joint acceleration: radians/second^2
- Cartesian translation: meters
- torques: N*m
- timestamps: stored explicitly with field names describing the time base
- depth units: described in metadata attrs and manifest fields

Camera calibration notes:

- RealSense native color/depth intrinsics are captured automatically
- resized intrinsics are also stored for the post-resize images used by the dataset
- RealSense depth-to-color and color-to-depth sensor extrinsics are captured automatically
- world/robot-frame extrinsics are task-configured in
  [fr3_dual_realsense.yml](/home/carl/deoxys_control/deoxys/config/data_tasks/fr3_dual_realsense.yml)
  and should be replaced with your actual calibration values

Additional metadata useful for imitation learning:

- camera acquisition settings record:
  - exposure
  - gain
  - white balance
  - auto-exposure state
  - auto-white-balance state
  - emitter / laser settings when available
  - depth preset when available

Units and field descriptions are written into:

- code comments in the data pipeline package
- `manifest.json`
- HDF5 dataset attrs such as `units`, `frame`, and `description`

## Useful Files in the Implementation

- CLI: [deoxys/deoxys/data/cli.py](/home/carl/deoxys_control/deoxys/deoxys/data/cli.py)
- Collector: [deoxys/deoxys/data/collector.py](/home/carl/deoxys_control/deoxys/deoxys/data/collector.py)
- HDF5 builder: [deoxys/deoxys/data/builder.py](/home/carl/deoxys_control/deoxys/deoxys/data/builder.py)
- Camera node: [deoxys/deoxys/data/camera_node.py](/home/carl/deoxys_control/deoxys/deoxys/data/camera_node.py)
- Viewer: [deoxys/deoxys/data/viewer.py](/home/carl/deoxys_control/deoxys/deoxys/data/viewer.py)
- Config loader: [deoxys/deoxys/data/config.py](/home/carl/deoxys_control/deoxys/deoxys/data/config.py)
- Metadata definitions: [deoxys/deoxys/data/metadata.py](/home/carl/deoxys_control/deoxys/deoxys/data/metadata.py)

## Troubleshooting

If `preflight` fails:

- make sure `deoxys.data up --task fr3_dual_realsense` is running
- make sure both cameras are physically connected
- make sure the camera serial numbers in the task config match the devices

If `view` opens but a stream is blank:

- confirm the camera serial mapping is correct
- confirm the camera is not already in use by another process

If `collect` saves very few samples:

- verify the robot state is streaming from the NUC
- verify you are producing real SpaceMouse motion after warmup
- check whether required streams are missing and being rejected

If `build` fails on mismatched lengths:

- inspect the `manifest.json` in the run folder
- inspect the raw NPZ files to see which stream was short
- re-collect the run if required streams were incomplete
