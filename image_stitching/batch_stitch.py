"""
Batch panorama generation: process all parquet files in dataset/camera_image/
and save one panorama per frame to outputs/.

Usage:
  python batch_stitch.py                         # first frame of every segment
  python batch_stitch.py --all-frames            # every frame of every segment
  python batch_stitch.py --output-dir my_output  # custom output folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from waymo_loader import (
    load_camera_images_from_parquet,
    load_camera_calibration,
    list_frames_in_parquet,
)
from cylindrical_stitch import build_cylindrical_panorama_fast


def process_parquet(
    parquet_path: Path,
    cal_path: Path,
    output_dir: Path,
    all_frames: bool,
) -> None:
    segment = parquet_path.stem
    print(f"\n[{segment}]")

    if not cal_path.is_file():
        print(f"  Skipping — no calibration file found at {cal_path}")
        return

    timestamps = list_frames_in_parquet(parquet_path)
    if not timestamps:
        print("  Skipping — no frames found.")
        return

    frames_to_run = timestamps if all_frames else [timestamps[0]]
    print(f"  {len(timestamps)} frames available, processing {len(frames_to_run)}")

    calibration = load_camera_calibration(cal_path)
    seg_dir = output_dir / segment
    seg_dir.mkdir(parents=True, exist_ok=True)

    for i, ts in enumerate(frames_to_run):
        out_path = seg_dir / f"frame_{i:04d}.png"
        if out_path.exists():
            print(f"  frame {i:04d} already exists, skipping.")
            continue

        images = load_camera_images_from_parquet(parquet_path, frame_timestamp_micros=ts)
        if not images:
            print(f"  frame {i:04d}: no images loaded, skipping.")
            continue

        panorama, cam_index_map = build_cylindrical_panorama_fast(images=images, calibration=calibration)
        cv2.imwrite(str(out_path), panorama)
        cammap_path = out_path.with_name(out_path.stem + "_cammap.png")
        cv2.imwrite(str(cammap_path), cam_index_map)
        print(f"  frame {i:04d} -> {out_path} + {cammap_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch stitch all Waymo parquet segments into panoramas."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_project_root / "dataset",
        help="Root dataset folder containing camera_image/ and camera_calibration/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "outputs",
        help="Folder to save panoramas (organized by segment).",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process every frame in each parquet (default: first frame only).",
    )
    args = parser.parse_args()

    image_dir = args.dataset_dir / "camera_image"
    cal_dir = args.dataset_dir / "camera_calibration"

    if not image_dir.is_dir():
        print(f"Error: camera_image folder not found at {image_dir}", file=sys.stderr)
        print("Make sure your dataset is placed at:", file=sys.stderr)
        print(f"  {args.dataset_dir}/camera_image/*.parquet", file=sys.stderr)
        print(f"  {args.dataset_dir}/camera_calibration/*.parquet", file=sys.stderr)
        return 1

    parquets = sorted(image_dir.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files found in {image_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(parquets)} segment(s) in {image_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {'all frames' if args.all_frames else 'first frame only'}")

    for parquet_path in parquets:
        cal_path = cal_dir / parquet_path.name
        process_parquet(parquet_path, cal_path, args.output_dir, args.all_frames)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
