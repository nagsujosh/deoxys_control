<p align="center">
<img src="./deoxys_github_logo.png">
</p>

<p align="center">
<a href="https://github.com/UT-Austin-RPL/deoxys_control/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/UT-Austin-RPL/deoxys_control/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/UT-Austin-RPL/deoxys_control" />
</a>
<a href="https://github.com/UT-Austin-RPL/deoxys_control/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/UT-Austin-RPL/deoxys_control?color=0088ff" />
</a>


[**[Documentation]**](https://zhuyifengzju.github.io/deoxys_docs/html/index.html) &ensp; 

Deoxys is a modular, real-time controller library for Franka Emika Panda arm, aiming to facilitate a wide range of robot learning research. Deoxys comes with a user-friendly python interface and real-time controller implementation in C++. If you are a [robosuite](https://github.com/ARISE-Initiative/robosuite) user, Deoxys APIs provide seamless transfer 
from you simulation codebase to real robot experiments!




https://user-images.githubusercontent.com/21077484/206338997-8dbaa128-dc63-4911-84ca-64d80a05673f.mp4



## Cite our codebase

If you use this codebase for your research projects, please cite our codebase based on the following project:

```
@article{zhu2022viola,
  title={VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors},
  author={Zhu, Yifeng and Joshi, Abhishek and Stone, Peter and Zhu, Yuke},
  journal={arXiv preprint arXiv:2210.11339},
  doi={10.48550/arXiv.2210.11339},
  year={2022}
}
```


# Installation of codebase

The setup has two machine-specific paths:
1. Desktop-side Python setup for the Deoxys client and teleoperation scripts
2. Intel NUC setup for the real-time Franka C++ interfaces

The steps below were validated on Ubuntu 22.04 with Python 3.10. For more background, see the [Codebase Installation Page](https://ut-austin-rpl.github.io/deoxys-docs/html/installation/codebase_installation.html).

Clone this repo to your robot workspace (for example `/home/USERNAME/robot-control-ws`) and move into the build directory:

```shell
cd deoxys_control/deoxys
```

## Desktop setup

Use a named Conda environment for the desktop machine. The examples below use `fr3`.

Create and activate the environment:

```shell
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n fr3 python=3.10 pip setuptools wheel -y
conda activate fr3
```

Generate the protobuf Python modules, install the Python requirements, and install Deoxys from this checkout:

```shell
cd deoxys_control/deoxys
make -j build_deoxys=1
pip install -U -r requirements.txt
pip install -e . --no-build-isolation
```

`make -j build_deoxys=1` requires a working Protobuf toolchain. If `cmake` reports that `protoc` or Protobuf is missing, run `LIBFRANKA_VERSION=0.9.0 ./InstallPackage` once first, or install a compatible Protobuf toolchain manually.

After these steps, the examples can be run directly from this checkout.

### SpaceMouse on Linux

For the 3Dconnexion SpaceMouse Wireless on Linux, install the bundled udev rule once:

```shell
./installation/create_spacemouse.sh
newgrp plugdev
```

Then start a fresh shell, reactivate `fr3`, and run the SpaceMouse example again.

## Franka Interface - Intel NUC

The NUC-side build expects the following source dependencies to exist inside `deoxys/`: `libfranka`, `zmqpp`, `yaml-cpp`, `spdlog`, and `protobuf`.

The helper script installs the required Ubuntu packages, clones those repositories, and installs Protobuf 3.13.0:

```shell
LIBFRANKA_VERSION=0.9.0 ./InstallPackage
```

The script requires internet access and `sudo`.

Then build the NUC-side binaries:

```shell
make -j build_franka=1
```

## A laundry list of pointers:
   - [How to turn on/off the robot](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/running_robots.html)
   - [How to install spacemouse](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html)
   - [How to set up the RTOS](https://ut-austin-rpl.github.io/deoxys-docs/html/installation/system_prerequisite.html)
   - [How to record and replay a trajectory](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/record_and_replay.html)
   - [How to write a simple motor program](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/handcrafting_motor_program.html)

# Control the robot

## Commands on Desktop

Here is a quick guide to run `Deoxys`.

On the desktop, activate `fr3`, move into `deoxys_control/deoxys`, and run:

```shell
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd deoxys_control/deoxys
python examples/run_deoxys_with_space_mouse.py 
```

Use the SpaceMouse right button or `Ctrl+C` to stop teleoperation cleanly.

For a no-hardware smoke test of the desktop-side Python setup, you can run:

```shell
python examples/run_deoxys_with_space_mouse.py --help
```

For the default 3Dconnexion SpaceMouse Wireless used during validation, the script uses vendor ID `9583` and product ID `50734`.

Change 1) SpaceMouse vendor_id and product_id ([here](https://github.com/UT-Austin-RPL/deoxys_control/blob/eb8d69f7f0838389fca81cac6b250ba05fc97f92/deoxys/examples/run_deoxys_with_space_mouse.py#L19)) 2) robot interface 
config ([here](https://github.com/UT-Austin-RPL/deoxys_control/blob/eb8d69f7f0838389fca81cac6b250ba05fc97f92/deoxys/examples/run_deoxys_with_space_mouse.py#L16)) if necessary.

You might also check and change the PC / NUC names [here](https://github.com/UT-Austin-RPL/deoxys_control/blob/master/deoxys/config/charmander.yml). 

## Commands on Control PC (Intel NUC)

Under `deoxys_control/deoxys`, run two commands. One for real-time control of the arm, one for non
real-time control of the gripper.

``` shell
bin/franka-interface config/charmander.yml
```

``` shell
bin/gripper-interface config/charmander.yml
```

# Teleop Data Pipeline

The repo now includes a structured data pipeline CLI for RGB-D camera bring-up, reset, teleoperation collection, and HDF5 dataset build. Saved color arrays are RGB in memory and on disk; the pipeline does not expose BGR-D datasets.

Full operator guide:

- [docs/teleop_data_pipeline.md](/home/carl/deoxys_control/docs/teleop_data_pipeline.md)
- [tools/calibration/README.md](/home/carl/deoxys_control/tools/calibration/README.md)

The calibration folder now includes scripts to:
- export RealSense-native intrinsics/extrinsics
- capture checkerboard calibration samples from the live Redis camera streams
- estimate a static camera extrinsic
- estimate a wrist-camera hand-eye extrinsic

The teleop guide now also documents how to create a new task YAML, how
`--task <name>` maps to that file, and exactly where raw `runN` folders and
`demo.hdf5` are written on disk. The CLI always uses the YAML filename stem for
task selection, while the YAML `name:` field controls the saved dataset path.

For shared lab use, the CLI now also includes task/date discovery helpers and a
task-template scaffold command so multiple researchers can stay inside the same
task-oriented storage layout:

```shell
deoxys.data tasks
deoxys.data dates --task fr3_dual_realsense
deoxys.data runs --task fr3_dual_realsense --all-dates
deoxys.data task-create --task bell_pepper_pick --from-task fr3_dual_realsense
```

## Default task and reset configs

The default dual-camera task config lives at `deoxys/config/data_tasks/fr3_dual_realsense.yml`.

The default home reset preset lives at `deoxys/config/data_resets/home_nominal.yml`.

The default task config also sets `default_reset_preset: home_nominal`, so
plain teleop, data collection, and replay all begin from the same nominal joint
configuration unless you override the preset explicitly.

These files are versioned and are intended to be copied or adapted per task so that controller settings, camera layout, resize policy, and reset behavior are reproducible.

## Scenario: New Task With a New Reset Preset

If you need a different home position for a specific experiment, the clean
workflow is:

1. Create a new task YAML from an existing template.
2. Create a new reset preset YAML.
3. Test that reset directly.
4. Point the new task at that reset as its default.
5. Use `teleop` or `collect` on the new task.

Example:

```shell
deoxys.data task-create --task bell_pepper_pick --from-task fr3_dual_realsense
```

Then create a new reset preset file such as
[home_bell_pepper_low.yml](/home/carl/deoxys_control/deoxys/config/data_resets/home_nominal.yml)
under [deoxys/config/data_resets](/home/carl/deoxys_control/deoxys/config/data_resets):

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

Test it directly:

```shell
deoxys.data reset --task bell_pepper_pick --preset home_bell_pepper_low
```

If it looks good, set the new task’s default reset in
[bell_pepper_pick.yml](/home/carl/deoxys_control/deoxys/config/data_tasks/fr3_dual_realsense.yml):

```yaml
default_reset_preset: home_bell_pepper_low
```

After that:

```shell
deoxys.data teleop --task bell_pepper_pick
deoxys.data collect --task bell_pepper_pick
```

will automatically use the new reset preset, while other tasks can keep using
their existing defaults.

## Data pipeline commands

Activate `fr3`, move into `deoxys_control`, and use the installed `deoxys.data` CLI:

```shell
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fr3
cd /home/carl/deoxys_control
```

Check Redis and camera stream freshness:

```shell
deoxys.data preflight --task fr3_dual_realsense
```

Start managed Redis and both camera nodes:

```shell
deoxys.data up --task fr3_dual_realsense
```

Inspect the task/date/run layout:

```shell
deoxys.data tasks
deoxys.data dates --task fr3_dual_realsense
deoxys.data runs --task fr3_dual_realsense --all-dates
```

Run the named reset preset:

```shell
deoxys.data reset --task fr3_dual_realsense --preset home_nominal
```

View one or both live Redis camera streams:

```shell
deoxys.data view --task fr3_dual_realsense --camera-role all
```

Run plain teleoperation without recording a dataset:

```shell
deoxys.data teleop --task fr3_dual_realsense
deoxys.data teleop --task fr3_dual_realsense --no-reset
deoxys.data teleop --task fr3_dual_realsense --inverse
```

Collect one teleoperation run:

```shell
deoxys.data collect --task fr3_dual_realsense
deoxys.data collect --task fr3_dual_realsense --no-reset
deoxys.data collect --task fr3_dual_realsense --inverse
```

Validate a raw run before HDF5 build, and optionally replay it:

```shell
deoxys.data validate --task fr3_dual_realsense --date 2026-03-15 --run run1
deoxys.data validate --task fr3_dual_realsense --date 2026-03-15 --run run1 --play
```

Build `demo.hdf5` from one task/date directory:

```shell
deoxys.data build --task fr3_dual_realsense --date 2026-03-14 --overwrite
```

Or aggregate several collection dates for the same task:

```shell
deoxys.data build --task fr3_dual_realsense --all-dates --overwrite
deoxys.data build --task fr3_dual_realsense --dates 2026-03-17 2026-03-18 --overwrite
```

Replay one stored raw run on the robot using its saved delta actions:

```shell
deoxys.data replay --task fr3_dual_realsense --date 2026-03-15 --run run1
deoxys.data replay --task fr3_dual_realsense --date 2026-03-15 --run run1 --no-reset
```

Export quick-look MP4 files from a built HDF5 dataset:

```shell
deoxys.data video --hdf5 /home/carl/deoxys_control/data/fr3_dual_realsense/2026-03-15/demo.hdf5
python tools/data/export_hdf5_demo_video.py --hdf5 /home/carl/deoxys_control/data/fr3_dual_realsense/2026-03-15/demo.hdf5
```

Add `--include-depth` if you want depth visualizations beside RGB. If depth is
requested but a demo or camera stream does not have depth, the exporter now
falls back to RGB for that stream and logs that depth was unavailable.

## Dataset contents

The new collector stores:

- RGB and optional depth for both fixed camera roles `agentview` and `wrist`
- default validated camera resolution is now `640x480` at `30 Hz` for both streams
- camera intrinsics and extrinsics metadata for downstream geometry-aware exports
- camera acquisition settings such as exposure, gain, white balance, and depth emitter settings when available
- one canonical delta-action stream saved as `testing_demo_delta_action.npz` in raw runs and `delta_actions` in HDF5
- 10D end-effector state features stored as xyz position plus 6D rotation plus gripper width
- per-camera validity masks so optional-camera placeholders and missing depth stay explicit
- per-camera calibration JSON so RGB-D geometry stays recoverable later without keeping large debug-side metadata archives

Current validated rates for the default task:

- robot state publisher: `100 Hz`
- policy / teleoperation command loop: `20 Hz`
- trajectory interpolation on the robot side: `500 Hz`
- camera capture and publish: `30 Hz` per camera
- the saved dataset therefore has a nominal action/state rate of about `20 Hz`, bounded by the policy loop rather than the camera frame rate

Units and conventions are documented in code comments, run manifests, and HDF5 dataset attributes wherever practical.

The pipeline now stores `ee_state_10d` as the main learning-facing pose-plus-gripper feature, and it does not store Euler roll/pitch/yaw orientation state features. The raw-run format is intentionally lean now: it keeps the tensors needed for RGB-D imitation learning plus calibration/validity metadata, and drops older debug-heavy arrays like observed deltas, tracking errors, sync archives, and large robot-metadata dumps from newly collected runs.

Collection behavior notes:

- The collector now keeps the imitation-learning pairing contract `obs_t -> delta_action_t -> control_t`, so observations are recorded before the current command is sent.
- The saved `delta_actions` dataset is the single canonical delta-action stream used for HDF5 export and raw-run replay.
- For the default OSC controllers, `delta_actions` stores the controller-space end-effector delta command sent to Franka/Deoxys after SpaceMouse mapping. It is not the raw SpaceMouse HID delta.
- The same task config now drives both plain `teleop` and dataset `collect`, so controller, interface, and SpaceMouse settings stay aligned.
- If the task config defines `default_reset_preset`, `teleop`, `collect`, and `replay` all run that preset first so they start from the same joint configuration, unless you pass `--no-reset`.
- Before plain teleop or collection becomes active, the pipeline explicitly opens the gripper and verifies it reached a near-fully-open width. If that cannot be confirmed, startup fails cleanly instead of beginning with a closed hand.
- Near-zero robot state reads can fall back to the previous valid state when `collection.state_zero_fallback: true`.
- Manual `Ctrl+C` during collection discards the run and removes the partial `runN` directory instead of saving partial files or metadata.
- The SpaceMouse right button finishes the current collection run and saves it if valid samples were recorded.
- The collector now emits a short terminal bell when it switches into active recording after the first valid motion, and a two-beep confirmation when the SpaceMouse right button is accepted as `finish and save`.
- When a run is discarded or cleaned up after failure, the CLI reports `run_dir: null` so it does not point at a deleted directory.
- Collection failures now return a structured `failure_reason` in CLI output.
- Failed runs are deleted by default with `collection.keep_failed_runs: false`.
  This setting applies to genuine collection failures, not user-canceled runs.
  User-canceled runs are always discarded and removed.
- Optional cameras can be declared with `collection.optional_camera_roles`; the collector inserts aligned placeholders and writes a validity mask instead of shrinking the run.
- Depth-enabled cameras also write a separate depth-validity mask, so zero-filled placeholder depth images are distinguishable from real depth frames.
- Use `--inverse` on `deoxys.data teleop` or `deoxys.data collect` for a temporary arm-motion inversion without editing the task YAML. Gripper semantics stay normal.
- Use `collection.action_multipliers` in the task YAML when you want a persistent custom inversion.

The `deoxys.data` commands are the canonical teleoperation/data-collection
workflow. Older example scripts under `deoxys/examples/` are still available as
references, but they keep their own save prompts and should not be treated as
the authoritative data-pipeline interface.

Action semantics are documented in one place in [docs/teleop_data_pipeline.md](/home/carl/deoxys_control/docs/teleop_data_pipeline.md), including the 7D action layout, gripper sign convention, controller frame, controller-side scaling, and the distinction between 6D orientation state features and the controller-native 3D rotational command channels.
