"""
Build the inpainting dataset (images/, masks/, masked/) from your stitching outputs.

No need to download the full Waymo training folder. This uses:
  image_stitching/outputs/<segment>/frame_*.png
  image_stitching/outputs/<segment>/frame_*_cammap.png

Output: one triplet per (frame, camera_index). Each camera failure is one sample.

Usage:
  python build_inpainting_dataset.py --output-dir inpainting/waymo_data/masks
  python build_inpainting_dataset.py --output-dir ./waymo_masks --segments image_stitching/outputs
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

CAM_NAMES = ["SIDE_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT"]


def main():
    ap = argparse.ArgumentParser(description="Build images/masks/masked from stitching outputs.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inpainting/waymo_data/masks"),
        help="Root folder to create images/, masks/, masked/ under.",
    )
    ap.add_argument(
        "--segments-dir",
        type=Path,
        default=Path("image_stitching/outputs"),
        help="Folder containing segment subfolders (each with frame_*.png and frame_*_cammap.png).",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames per segment (default: all).",
    )
    ap.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Max segments to process (default: all).",
    )
    args = ap.parse_args()

    out_root = args.output_dir.resolve()
    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_masked = out_root / "masked"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_masked.mkdir(parents=True, exist_ok=True)

    segments_dir = args.segments_dir.resolve()
    if not segments_dir.is_dir():
        print(f"Error: segments dir not found: {segments_dir}")
        return 1

    segment_dirs = sorted([d for d in segments_dir.iterdir() if d.is_dir()])
    if args.max_segments is not None:
        segment_dirs = segment_dirs[: args.max_segments]

    total = 0
    for seg_dir in segment_dirs:
        frames = sorted(seg_dir.glob("frame_*.png"))
        frames = [f for f in frames if "_cammap" not in f.name]
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        for frame_path in frames:
            stem = frame_path.stem  # e.g. frame_0000
            cammap_path = frame_path.parent / f"{stem}_cammap.png"
            if not cammap_path.is_file():
                print(f"Skip (no cammap): {frame_path}")
                continue
            panorama = cv2.imread(str(frame_path))
            cammap = cv2.imread(str(cammap_path), cv2.IMREAD_GRAYSCALE)
            if panorama is None or cammap is None:
                print(f"Skip (read failed): {frame_path}")
                continue
            for cam_id in range(5):
                name = f"{stem}_cam{cam_id}"
                mask = (cammap == cam_id).astype(np.uint8) * 255
                masked = panorama.copy()
                masked[cammap == cam_id] = 0
                cv2.imwrite(str(out_images / f"{name}.png"), panorama)
                cv2.imwrite(str(out_masks / f"{name}.png"), mask)
                cv2.imwrite(str(out_masked / f"{name}.png"), masked)
                total += 1
        print(f"  {seg_dir.name}: {len(frames)} frames -> {len(frames)*5} triplets")
    print(f"Done. Total triplets: {total}")
    print(f"  {out_images}")
    print(f"  {out_masks}")
    print(f"  {out_masked}")
    print("Use --root", str(out_root), "in eval_edgeconnect.py / train scripts.")
    return 0


if __name__ == "__main__":
    exit(main())
