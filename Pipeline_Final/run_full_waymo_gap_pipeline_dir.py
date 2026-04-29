#!/usr/bin/env python3
"""
run_full_waymo_gap_pipeline_dir.py

Directory-based full pipeline runner.

This version scans:
  --camera_image_dir
  --camera_calibration_dir

It pairs matching parquet files by filename, then runs the full pipeline over:
  segment parquet -> frame index -> stitch -> mask -> LaMa -> YOLO comparison.

This solves the issue where the previous script only reused one fixed parquet file.

Required stage scripts in current folder by default:
  - waymo_flat360_intrinsic_warp_v5_tight.py
  - mask_vertical_camera_seams.py
  - stage3_lama_inpaint_single.py
  - stage4_yolo_compare.py
  - waymo_loader.py

Example:
  python run_full_waymo_gap_pipeline_dir.py \
    --camera_image_dir /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/Dataset/dataset/camera_image \
    --camera_calibration_dir /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/Dataset/dataset/camera_calibration \
    --output_root full_pipeline_output_dir \
    --max_segments 3 \
    --frames_per_segment 2 \
    --lama_root /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/lama_official \
    --lama_model_path /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/lama_official/big-lama \
    --lama_checkpoint best.ckpt \
    --lama_gpu -1 \
    --yolo_model yolo11n.pt \
    --yolo_device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    print("\n[RUN]")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parquet_stem(path: Path) -> str:
    return path.stem


def safe_segment_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name


def list_parquets(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*.parquet"))
    return {p.name: p for p in files}


def pair_parquets(camera_image_dir: Path, camera_calibration_dir: Path) -> List[Tuple[str, Path, Path]]:
    image_files = list_parquets(camera_image_dir)
    calib_files = list_parquets(camera_calibration_dir)

    common = sorted(set(image_files.keys()) & set(calib_files.keys()))

    missing_calib = sorted(set(image_files.keys()) - set(calib_files.keys()))
    missing_image = sorted(set(calib_files.keys()) - set(image_files.keys()))

    if missing_calib:
        print("[WARN] Camera image files without matching calibration:")
        for f in missing_calib[:10]:
            print(f"  {f}")
        if len(missing_calib) > 10:
            print(f"  ... {len(missing_calib) - 10} more")

    if missing_image:
        print("[WARN] Calibration files without matching camera image:")
        for f in missing_image[:10]:
            print(f"  {f}")
        if len(missing_image) > 10:
            print(f"  ... {len(missing_image) - 10} more")

    pairs = [(name, image_files[name], calib_files[name]) for name in common]
    return pairs


def find_one(pattern: str, folder: Path, label: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {label} with pattern {pattern} in {folder}")
    return matches[0]


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        print(f"[WARN] No rows to write: {out_path}")
        return

    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def read_csv_rows(path: Path, extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = dict(r)
            out.update(extra)
            rows.append(out)
    return rows


def aggregate_yolo_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        if r.get("class_name") != "__ALL__":
            continue
        variant = r.get("variant", "")
        groups.setdefault(variant, []).append(r)

    out = []

    for variant, items in sorted(groups.items()):
        counts = [float(r.get("count", 0) or 0) for r in items]
        confs = [float(r.get("mean_confidence", 0) or 0) for r in items]

        out.append(
            {
                "variant": variant,
                "num_samples": len(items),
                "total_detections": int(sum(counts)),
                "mean_detections_per_sample": sum(counts) / len(counts) if counts else 0.0,
                "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera_image_dir", required=True)
    parser.add_argument("--camera_calibration_dir", required=True)
    parser.add_argument("--output_root", default="full_pipeline_output_dir")

    parser.add_argument("--max_segments", type=int, default=0, help="0 = all paired parquet segments.")
    parser.add_argument("--start_segment", type=int, default=0)
    parser.add_argument("--frames_per_segment", type=int, default=1)
    parser.add_argument("--start_frame", type=int, default=0)

    parser.add_argument("--stage1_script", default="waymo_flat360_intrinsic_warp_v5_tight.py")
    parser.add_argument("--stage2_script", default="mask_vertical_camera_seams.py")
    parser.add_argument("--stage3_script", default="stage3_lama_inpaint_single.py")
    parser.add_argument("--stage4_script", default="stage4_yolo_compare.py")

    parser.add_argument("--panorama_width", type=int, default=4800)
    parser.add_argument("--panorama_height", type=int, default=900)
    parser.add_argument("--vertical_fov_deg", type=float, default=55.0)
    parser.add_argument("--horizon", type=float, default=0.52)
    parser.add_argument("--sample_step", type=int, default=2)
    parser.add_argument("--splat_radius", type=int, default=1)
    parser.add_argument("--crop_pad_x", type=int, default=0)
    parser.add_argument("--crop_pad_y", type=int, default=0)

    parser.add_argument("--mask_percent", type=float, default=0.5)
    parser.add_argument("--max_seam_width_px", type=int, default=80)
    parser.add_argument("--full_height", type=int, default=1)

    parser.add_argument("--lama_root", required=True)
    parser.add_argument("--lama_model_path", required=True)
    parser.add_argument("--lama_checkpoint", default="best.ckpt")
    parser.add_argument("--lama_gpu", type=int, default=-1)
    parser.add_argument("--lama_refine", action="store_true")

    parser.add_argument("--yolo_model", default="yolo11n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.70)
    parser.add_argument("--yolo_imgsz", type=int, default=1280)
    parser.add_argument("--yolo_device", default="")
    parser.add_argument("--save_yolo_crops", action="store_true")

    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    camera_image_dir = Path(args.camera_image_dir)
    camera_calibration_dir = Path(args.camera_calibration_dir)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    for folder, label in [
        (camera_image_dir, "camera image directory"),
        (camera_calibration_dir, "camera calibration directory"),
    ]:
        if not folder.exists():
            raise FileNotFoundError(f"Missing {label}: {folder}")

    for script_path in [args.stage1_script, args.stage2_script, args.stage3_script, args.stage4_script]:
        if not Path(script_path).exists():
            raise FileNotFoundError(f"Missing stage script: {script_path}")

    pairs = pair_parquets(camera_image_dir, camera_calibration_dir)

    if not pairs:
        raise RuntimeError("No matching parquet pairs found. Filenames must match between directories.")

    pairs = pairs[args.start_segment:]

    if args.max_segments > 0:
        pairs = pairs[: args.max_segments]

    print(f"[INFO] Found paired parquet segments: {len(pairs)}")

    manifest_rows: List[Dict[str, Any]] = []
    all_summary_rows: List[Dict[str, Any]] = []
    all_detection_rows: List[Dict[str, Any]] = []

    for seg_idx, (filename, image_parquet, calib_parquet) in enumerate(pairs, start=args.start_segment):
        segment_name = safe_segment_name(image_parquet)
        print("\n" + "=" * 100)
        print(f"[SEGMENT {seg_idx}] {filename}")
        print("=" * 100)

        for frame_index in range(args.start_frame, args.start_frame + args.frames_per_segment):
            print("\n" + "-" * 80)
            print(f"[SAMPLE] segment={segment_name} frame_index={frame_index}")
            print("-" * 80)

            sample_id = f"{segment_name}_frame_{frame_index:06d}"
            sample_dir = output_root / sample_id

            stitch_dir = sample_dir / "stage1_stitch"
            mask_dir = sample_dir / "stage2_mask"
            lama_dir = sample_dir / "stage3_lama"
            yolo_dir = sample_dir / "stage4_yolo"

            for d in [stitch_dir, mask_dir, lama_dir, yolo_dir]:
                ensure_dir(d)

            # Stage 1
            if not (args.skip_existing and list(stitch_dir.glob("*_flat360_intrinsic_warp_v5_tight.png"))):
                cmd = [
                    args.python_exe,
                    args.stage1_script,
                    "--camera_image_parquet", str(image_parquet),
                    "--camera_calibration_parquet", str(calib_parquet),
                    "--frame_index", str(frame_index),
                    "--output_dir", str(stitch_dir),
                    "--panorama_width", str(args.panorama_width),
                    "--panorama_height", str(args.panorama_height),
                    "--vertical_fov_deg", str(args.vertical_fov_deg),
                    "--horizon", str(args.horizon),
                    "--sample_step", str(args.sample_step),
                    "--splat_radius", str(args.splat_radius),
                    "--crop_to_content",
                    "--crop_pad_x", str(args.crop_pad_x),
                    "--crop_pad_y", str(args.crop_pad_y),
                ]
                run_cmd(cmd)
            else:
                print("[SKIP] Stage 1 exists")

            original = find_one("*_flat360_intrinsic_warp_v5_tight.png", stitch_dir, "stitched panorama")
            camera_id_map = find_one("*_flat360_intrinsic_warp_v5_tight_camera_id_map.png", stitch_dir, "camera ID map")

            # Stage 2
            if not (args.skip_existing and list(mask_dir.glob("*_vertical_seam_mask_p*.png"))):
                cmd = [
                    args.python_exe,
                    args.stage2_script,
                    "--panorama", str(original),
                    "--camera_id_map", str(camera_id_map),
                    "--output_dir", str(mask_dir),
                    "--mask_percent", str(args.mask_percent),
                    "--max_seam_width_px", str(args.max_seam_width_px),
                    "--full_height", str(args.full_height),
                ]
                run_cmd(cmd)
            else:
                print("[SKIP] Stage 2 exists")

            mask = find_one("*_vertical_seam_mask_p*.png", mask_dir, "inpaint mask")
            masked = find_one("*_vertical_seam_masked_p*.png", mask_dir, "masked panorama")

            # Stage 3
            if not (args.skip_existing and list(lama_dir.glob("*_lama_inpainted.png"))):
                cmd = [
                    args.python_exe,
                    args.stage3_script,
                    "--image", str(original),
                    "--mask", str(mask),
                    "--lama_root", args.lama_root,
                    "--model_path", args.lama_model_path,
                    "--checkpoint", args.lama_checkpoint,
                    "--gpu", str(args.lama_gpu),
                    "--output_dir", str(lama_dir),
                    "--temp_input_dir", str(sample_dir / "stage3_lama_temp_input"),
                    "--sample_name", sample_id,
                ]
                if args.lama_refine:
                    cmd.append("--refine")
                run_cmd(cmd)
            else:
                print("[SKIP] Stage 3 exists")

            inpainted = find_one("*_lama_inpainted.png", lama_dir, "inpainted panorama")

            # Stage 4
            if not (args.skip_existing and (yolo_dir / "yolo_summary.csv").exists()):
                cmd = [
                    args.python_exe,
                    args.stage4_script,
                    "--original", str(original),
                    "--masked", str(masked),
                    "--inpainted", str(inpainted),
                    "--model", args.yolo_model,
                    "--output_dir", str(yolo_dir),
                    "--conf", str(args.yolo_conf),
                    "--iou", str(args.yolo_iou),
                    "--imgsz", str(args.yolo_imgsz),
                ]
                if args.yolo_device:
                    cmd.extend(["--device", args.yolo_device])
                if args.save_yolo_crops:
                    cmd.append("--save_crops")
                run_cmd(cmd)
            else:
                print("[SKIP] Stage 4 exists")

            summary_csv = yolo_dir / "yolo_summary.csv"
            detections_csv = yolo_dir / "yolo_detections.csv"

            extra = {
                "segment_index": seg_idx,
                "segment_file": filename,
                "frame_index": frame_index,
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
            }

            all_summary_rows.extend(read_csv_rows(summary_csv, extra))
            all_detection_rows.extend(read_csv_rows(detections_csv, extra))

            manifest_rows.append(
                {
                    **extra,
                    "camera_image_parquet": str(image_parquet),
                    "camera_calibration_parquet": str(calib_parquet),
                    "original": str(original),
                    "camera_id_map": str(camera_id_map),
                    "mask": str(mask),
                    "masked": str(masked),
                    "inpainted": str(inpainted),
                    "yolo_summary": str(summary_csv),
                    "yolo_detections": str(detections_csv),
                }
            )

    manifest_csv = output_root / "pipeline_manifest.csv"
    summary_csv = output_root / "all_yolo_summary_rows.csv"
    detections_csv = output_root / "all_yolo_detections.csv"
    aggregate_csv = output_root / "aggregate_yolo_counts.csv"
    manifest_json = output_root / "pipeline_manifest.json"

    write_csv(manifest_rows, manifest_csv)
    write_csv(all_summary_rows, summary_csv)
    write_csv(all_detection_rows, detections_csv)

    aggregate_rows = aggregate_yolo_summary(all_summary_rows)
    write_csv(aggregate_rows, aggregate_csv)

    with open(manifest_json, "w") as f:
        json.dump(manifest_rows, f, indent=2)

    print("\n" + "=" * 100)
    print("[FULL DIRECTORY PIPELINE DONE]")
    print("=" * 100)
    print(f"Manifest:       {manifest_csv}")
    print(f"YOLO summaries: {summary_csv}")
    print(f"YOLO detections:{detections_csv}")
    print(f"Aggregate:      {aggregate_csv}")
    print(f"Manifest JSON:  {manifest_json}")

    print("\n[AGGREGATE]")
    for row in aggregate_rows:
        print(
            f'{row["variant"]:9s}: total={row["total_detections"]}, '
            f'mean/sample={row["mean_detections_per_sample"]:.2f}, '
            f'mean_conf={row["mean_confidence"]:.3f}'
        )


if __name__ == "__main__":
    main()
