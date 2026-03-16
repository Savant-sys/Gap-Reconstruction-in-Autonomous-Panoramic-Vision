"""
Run the image stitching pipeline on a single frame from the Waymo Open Dataset.

Loads one synchronized multi-camera frame from a camera_image parquet file,
orders the cameras for a left-to-right panorama, runs feature-based stitching,
and saves the result. Use this script as the entry point for generating
panoramas that can later be processed (e.g. region removal + inpainting).

Example:
  py run_stitch.py --parquet dataset/camera_image/8993680275027614595_2520_000_2540_000.parquet --output panorama.png
  py run_stitch.py --parquet dataset/camera_image/8993680275027614595_2520_000_2540_000.parquet --frame-index 0 --descriptor SIFT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root so imports work when run from any directory
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from waymo_loader import (
    load_camera_images_from_parquet,
    load_camera_calibration,
    list_frames_in_parquet,
    get_panorama_order,
)
from image_stitcher import PanoramaStitcher
from cylindrical_stitch import build_cylindrical_panorama_fast


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stitch Waymo multi-camera frame into a panorama."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to a camera_image parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("panorama_output.png"),
        help="Output path for the stitched panorama image.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index of the frame to use (0 = first timestamp in the file).",
    )
    parser.add_argument(
        "--frame-timestamp",
        type=int,
        default=None,
        help="Use this frame timestamp (micros) instead of --frame-index.",
    )
    parser.add_argument(
        "--descriptor",
        type=str,
        choices=["ORB", "SIFT"],
        default="ORB",
        help="Feature descriptor for matching (ORB is faster, SIFT often more robust).",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="Minimum number of inlier matches required between adjacent images.",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="Blend overlapping regions instead of overwriting.",
    )
    parser.add_argument(
        "--list-frames",
        action="store_true",
        help="Only list frame timestamps in the parquet and exit.",
    )
    parser.add_argument(
        "--save-inputs",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save the 5 separate camera images (and a strip image) to this folder so you can show input vs output.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["calibration", "homography"],
        default="calibration",
        help="Stitching method: calibration (uses camera_calibration for cylindrical panorama, recommended) or homography (feature-based).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to camera_calibration parquet (default: same filename as --parquet in dataset/camera_calibration/).",
    )
    args = parser.parse_args()

    parquet_path = args.parquet
    if not parquet_path.is_file():
        print(f"Error: parquet file not found: {parquet_path}", file=sys.stderr)
        # Suggest actual files if camera_image folder exists
        camera_dir = _project_root / "dataset" / "camera_image"
        if camera_dir.is_dir():
            parquets = list(camera_dir.glob("*.parquet"))[:5]
            if parquets:
                print("Example paths (use one of these):", file=sys.stderr)
                for p in parquets:
                    print(f"  --parquet {p}", file=sys.stderr)
        return 1

    if args.list_frames:
        timestamps = list_frames_in_parquet(parquet_path)
        print(f"Frames in {parquet_path}: {len(timestamps)}")
        for i, ts in enumerate(timestamps[:20]):
            print(f"  [{i}] {ts}")
        if len(timestamps) > 20:
            print(f"  ... and {len(timestamps) - 20} more")
        return 0

    # Resolve frame timestamp
    timestamps = list_frames_in_parquet(parquet_path)
    if not timestamps:
        print("Error: no frames in parquet file.", file=sys.stderr)
        return 1
    if args.frame_timestamp is not None:
        if args.frame_timestamp not in timestamps:
            print(f"Error: timestamp {args.frame_timestamp} not in file.", file=sys.stderr)
            return 1
        frame_ts = args.frame_timestamp
    else:
        idx = args.frame_index
        if idx < 0 or idx >= len(timestamps):
            print(f"Error: frame-index must be in [0, {len(timestamps)-1}].", file=sys.stderr)
            return 1
        frame_ts = timestamps[idx]

    # Load all cameras for this frame
    print(f"Loading frame timestamp {frame_ts} from {parquet_path} ...")
    frames = load_camera_images_from_parquet(parquet_path, frame_timestamp_micros=frame_ts)
    if not frames:
        print("Error: no images loaded for this frame.", file=sys.stderr)
        return 1

    order = get_panorama_order()
    images = []
    for name in order:
        if name in frames:
            images.append(frames[name])
        else:
            print(f"Warning: camera {name} not in parquet, skipping.", file=sys.stderr)
    import cv2

    # Optionally save the 5 separate input images (and a horizontal strip) for "input vs output"
    if args.save_inputs is not None:
        out_dir = args.save_inputs
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in order:
            if name in frames:
                p = out_dir / f"{name}.png"
                cv2.imwrite(str(p), frames[name])
                print(f"Saved input image: {p}")
        # Build strip: cameras can have different resolutions, so resize to common height
        imgs = [frames[n] for n in order if n in frames]
        h_max = max(im.shape[0] for im in imgs)
        resized = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max)) for im in imgs]
        strip = np.hstack(resized)
        strip_path = out_dir / "inputs_strip.png"
        cv2.imwrite(str(strip_path), strip)
        print(f"Saved input strip (all 5 in order): {strip_path}")

    if args.method == "calibration":
        # Calibration-based cylindrical panorama (recommended)
        cal_path = args.calibration
        if cal_path is None:
            # Same filename as parquet, in dataset/camera_calibration/
            cal_path = _project_root / "dataset" / "camera_calibration" / parquet_path.name
        if not cal_path.is_file():
            print(f"Error: calibration file not found: {cal_path}", file=sys.stderr)
            print("Use --calibration PATH or place the parquet in dataset/camera_calibration/ with the same filename.", file=sys.stderr)
            return 1
        print(f"Loading calibration from {cal_path} ...")
        calibration = load_camera_calibration(cal_path)
        print(f"Stitching {len(frames)} images with calibration (cylindrical) ...")
        panorama = build_cylindrical_panorama_fast(
            images=frames,
            calibration=calibration,
            out_width=4000,
            out_height=1200,
        )
    else:
        # Homography-based (feature matching)
        if len(images) < 2:
            print("Error: need at least two camera images to stitch.", file=sys.stderr)
            return 1
        print(f"Stitching {len(images)} images in order: {[n for n in order if n in frames]} (homography) ...")
        stitcher = PanoramaStitcher(
            descriptor=args.descriptor,
            min_match_count=args.min_matches,
            blend=args.blend,
        )
        panorama = stitcher.stitch(images)

    if panorama is None:
        print("Error: stitching failed.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), panorama)
    print(f"Saved panorama to {args.output} (shape {panorama.shape}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
