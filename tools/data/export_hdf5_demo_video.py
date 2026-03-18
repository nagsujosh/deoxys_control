#!/usr/bin/env python3
"""Export quick-look MP4 videos from a Deoxys `demo.hdf5` file."""

from __future__ import annotations

import argparse
import json

from deoxys.data.video_export import export_hdf5_demo_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one MP4 per demo group from a Deoxys demo.hdf5 file."
    )
    parser.add_argument("--hdf5", required=True, help="Path to demo.hdf5")
    parser.add_argument("--output-dir", help="Directory where MP4 files are written")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video frame rate")
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Append available depth visualizations next to RGB frames",
    )
    parser.add_argument(
        "--demo-indices",
        nargs="*",
        type=int,
        help="Optional list of demo indices to export, for example: --demo-indices 0 3 4",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_paths = export_hdf5_demo_videos(
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        demo_indices=args.demo_indices,
        fps=args.fps,
        include_depth=args.include_depth,
    )
    print(json.dumps([str(path) for path in output_paths], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
