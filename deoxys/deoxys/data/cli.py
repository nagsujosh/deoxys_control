"""Operator-facing CLI for the teleoperation data pipeline."""

from __future__ import annotations

import argparse
import json
import time

from .config import (
    create_task_config_from_template,
    list_task_config_stems,
    load_reset_preset,
    load_reset_preset_from_path,
    load_task_config,
    load_task_config_from_path,
)
from .logging_utils import get_data_logger
from .paths import get_task_root
from .process_manager import ProcessManager
from .run_tools import list_task_dates, list_task_runs

logger = get_data_logger("cli")


def _resolve_optional_preset(
    task_cfg,
    preset_name: str | None,
    *,
    preset_path: str | None = None,
    no_reset: bool = False,
):
    """Resolve a CLI preset override or fall back to the task default."""

    if no_reset:
        return None
    if preset_name and preset_path:
        raise ValueError("Pass only one of --preset or --preset-path.")
    if preset_path:
        return load_reset_preset_from_path(preset_path)
    effective_name = preset_name or task_cfg.default_reset_preset
    if not effective_name:
        return None
    return load_reset_preset(effective_name)


def _add_task_config_selector(command: argparse.ArgumentParser) -> None:
    """Add the shared `--task` / `--config` selector for task-backed commands."""

    command.add_argument("--task")
    command.add_argument("--config")


def _load_task_cfg_from_args(args) -> object:
    """Resolve a task config from either a shared profile stem or explicit YAML path."""

    if bool(getattr(args, "task", None)) == bool(getattr(args, "config", None)):
        raise ValueError("Exactly one of --task or --config is required.")
    if getattr(args, "config", None):
        return load_task_config_from_path(args.config)
    return load_task_config(args.task)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deoxys.data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("tasks")

    task_create_parser = subparsers.add_parser("task-create")
    task_create_parser.add_argument("--task", required=True)
    task_create_parser.add_argument("--from-task", default="fr3_dual_realsense")
    task_create_parser.add_argument("--dataset-name")
    task_create_parser.add_argument("--output-root")
    task_create_parser.add_argument("--overwrite", action="store_true")

    dates_parser = subparsers.add_parser("dates")
    dates_parser.add_argument("--task", required=True)

    runs_parser = subparsers.add_parser("runs")
    runs_parser.add_argument("--task", required=True)
    runs_group = runs_parser.add_mutually_exclusive_group()
    runs_group.add_argument("--date")
    runs_group.add_argument("--all-dates", action="store_true")

    for command_name in ("preflight", "up", "view", "teleop"):
        command = subparsers.add_parser(command_name)
        _add_task_config_selector(command)

    reset_parser = subparsers.add_parser("reset")
    _add_task_config_selector(reset_parser)
    reset_parser.add_argument("--preset")
    reset_parser.add_argument("--preset-path")

    collect_parser = subparsers.add_parser("collect")
    _add_task_config_selector(collect_parser)
    collect_parser.add_argument("--preset")
    collect_parser.add_argument("--preset-path")

    replay_parser = subparsers.add_parser("replay")
    _add_task_config_selector(replay_parser)
    replay_parser.add_argument("--date")
    replay_parser.add_argument("--run")
    replay_parser.add_argument("--run-dir")
    replay_parser.add_argument("--preset")
    replay_parser.add_argument("--preset-path")
    replay_parser.add_argument("--max-steps", type=int, default=0)

    build_parser_cmd = subparsers.add_parser("build")
    _add_task_config_selector(build_parser_cmd)
    build_group = build_parser_cmd.add_mutually_exclusive_group()
    build_group.add_argument("--date")
    build_group.add_argument("--dates", nargs="+")
    build_group.add_argument("--all-dates", action="store_true")
    build_parser_cmd.add_argument("--output-hdf5")
    build_parser_cmd.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    _add_task_config_selector(validate_parser)
    validate_parser.add_argument("--date")
    validate_parser.add_argument("--run")
    validate_parser.add_argument("--run-dir")
    validate_parser.add_argument("--play", action="store_true")
    validate_parser.add_argument("--fps", type=float, default=10.0)
    validate_parser.add_argument("--include-depth", action="store_true")

    video_parser = subparsers.add_parser("video")
    video_parser.add_argument("--hdf5", required=True)
    video_parser.add_argument("--output-dir")
    video_parser.add_argument("--fps", type=float, default=20.0)
    video_parser.add_argument("--include-depth", action="store_true")
    video_parser.add_argument("--demo-indices", nargs="*", type=int)

    view_parser = subparsers.choices["view"]
    view_parser.add_argument("--camera-role", choices=["agentview", "wrist", "all"], default="all")
    teleop_parser = subparsers.choices["teleop"]
    teleop_parser.add_argument("--preset")
    teleop_parser.add_argument("--preset-path")
    teleop_parser.add_argument("--no-reset", action="store_true")
    teleop_parser.add_argument("--max-steps", type=int, default=0)
    teleop_parser.add_argument("--inverse", action="store_true")
    collect_parser.add_argument("--no-reset", action="store_true")
    collect_parser.add_argument("--inverse", action="store_true")
    replay_parser.add_argument("--no-reset", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"tasks", "task-create", "video"}:
        pass
    elif args.command in {"dates", "runs"}:
        if not args.task:
            parser.error("--task is required")
    else:
        try:
            _load_task_cfg_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "tasks":
        task_summaries = []
        for stem in list_task_config_stems():
            task_cfg = load_task_config(stem)
            dates = list_task_dates(task_cfg)
            task_summaries.append(
                {
                    "task": stem,
                    "dataset_name": task_cfg.name,
                    "config_path": task_cfg.config_path,
                    "task_root": str(get_task_root(task_cfg.output_root, task_cfg.name)),
                    "num_dates": len(dates),
                    "num_runs": sum(int(item["run_count"]) for item in dates),
                    "latest_date": (dates[-1]["date"] if dates else None),
                }
            )
        print(json.dumps({"tasks": task_summaries}, indent=2, sort_keys=True))
        return 0

    if args.command == "task-create":
        result = create_task_config_from_template(
            args.task,
            template_task=args.from_task,
            dataset_name=args.dataset_name,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "dates":
        task_cfg = load_task_config(args.task)
        print(json.dumps({"task": args.task, "dates": list_task_dates(task_cfg)}, indent=2, sort_keys=True))
        return 0

    if args.command == "runs":
        task_cfg = load_task_config(args.task)
        print(
            json.dumps(
                {
                    "task": args.task,
                    "runs": list_task_runs(
                        task_cfg,
                        date_str=args.date,
                        all_dates=args.all_dates,
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "preflight":
        task_cfg = _load_task_cfg_from_args(args)
        report = ProcessManager(task_cfg).health_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        optional_roles = set(task_cfg.collection.optional_camera_roles)
        cameras_healthy = all(
            (
                camera.role in optional_roles
                or report.get(camera.role, {}).get("fresh", False)
            )
            for camera in task_cfg.cameras
        )
        logger.info(
            "Preflight for task `%s`: redis=%s cameras=%s",
            args.task,
            report.get("redis", {}).get("reachable"),
            cameras_healthy,
        )
        return 0 if report.get("redis", {}).get("reachable") and cameras_healthy else 1

    if args.command == "up":
        task_cfg = _load_task_cfg_from_args(args)
        manager = ProcessManager(task_cfg)
        manager.start_redis()
        manager.start_camera_nodes()
        try:
            while True:
                print(json.dumps(manager.health_report(), indent=2, sort_keys=True))
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Stopping managed pipeline services for task `%s`", args.task)
            manager.stop_all()
        return 0

    if args.command == "reset":
        from .reset import run_reset

        task_cfg = _load_task_cfg_from_args(args)
        if not args.preset and not args.preset_path:
            parser.error("reset requires one of --preset or --preset-path")
        if args.preset and args.preset_path:
            parser.error("reset accepts only one of --preset or --preset-path")
        preset = load_reset_preset_from_path(args.preset_path) if args.preset_path else load_reset_preset(args.preset)
        print(json.dumps(run_reset(task_cfg, preset), indent=2, sort_keys=True))
        return 0

    if args.command == "view":
        from .viewer import run_viewer

        task_cfg = _load_task_cfg_from_args(args)
        roles = task_cfg.camera_roles if args.camera_role == "all" else [args.camera_role]
        run_viewer(task_cfg, roles)
        return 0

    if args.command == "teleop":
        from .teleop import run_teleop

        task_cfg = _load_task_cfg_from_args(args)
        preset = _resolve_optional_preset(
            task_cfg,
            args.preset,
            preset_path=args.preset_path,
            no_reset=args.no_reset,
        )
        result = run_teleop(
            task_cfg,
            preset=preset,
            max_steps=args.max_steps,
            inverse=args.inverse,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"success", "stopped", "interrupted"} else 1

    if args.command == "collect":
        from .collector import TeleopCollector

        task_cfg = _load_task_cfg_from_args(args)
        preset = _resolve_optional_preset(
            task_cfg,
            args.preset,
            preset_path=args.preset_path,
            no_reset=args.no_reset,
        )
        result = TeleopCollector(task_cfg, inverse=args.inverse).collect(
            preset=preset,
        )
        print(
            json.dumps(
                {
                    "run_dir": None if result.run_dir is None else str(result.run_dir),
                    "status": result.status,
                    "num_samples": result.num_samples,
                    "failure_reason": result.failure_reason,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.status == "success" else 1

    if args.command == "build":
        from .builder import HDF5DatasetBuilder

        task_cfg = _load_task_cfg_from_args(args)
        output_path = HDF5DatasetBuilder(task_cfg).build(
            date_str=args.date,
            date_strs=args.dates,
            all_dates=args.all_dates,
            overwrite=args.overwrite,
            output_path=args.output_hdf5,
        )
        print(str(output_path))
        return 0

    if args.command == "replay":
        from .replay import replay_run

        task_cfg = _load_task_cfg_from_args(args)
        preset = _resolve_optional_preset(
            task_cfg,
            args.preset,
            preset_path=args.preset_path,
            no_reset=args.no_reset,
        )
        result = replay_run(
            task_cfg,
            date_str=args.date,
            run_name=args.run,
            run_dir=args.run_dir,
            preset=preset,
            max_steps=args.max_steps,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"success", "interrupted"} else 1

    if args.command == "validate":
        from .validate import validate_run

        task_cfg = _load_task_cfg_from_args(args)
        report = validate_run(
            task_cfg,
            date_str=args.date,
            run_name=args.run,
            run_dir=args.run_dir,
            play=args.play,
            fps=args.fps,
            include_depth=args.include_depth,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1

    if args.command == "video":
        from .video_export import export_hdf5_demo_videos

        output_paths = export_hdf5_demo_videos(
            hdf5_path=args.hdf5,
            output_dir=args.output_dir,
            demo_indices=args.demo_indices,
            fps=args.fps,
            include_depth=args.include_depth,
        )
        print(json.dumps([str(path) for path in output_paths], indent=2))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
