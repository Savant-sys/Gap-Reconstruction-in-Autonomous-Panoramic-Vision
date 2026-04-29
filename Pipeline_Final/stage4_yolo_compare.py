#!/usr/bin/env python3
"""
stage4_yolo_compare.py

Stage 4: Run YOLO object detection on:
  1. original stitched panorama
  2. masked stitched panorama
  3. inpainted stitched panorama

Outputs:
  - annotated images
  - per-image detection CSV
  - summary CSV
  - JSON with all detections

Install:
  python -m pip install ultralytics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def run_yolo_on_image(
    model,
    image_path: Path,
    variant_name: str,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    save_crops: bool,
) -> List[Dict[str, Any]]:
    kwargs = {
        "source": str(image_path),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }

    if device is not None and device != "":
        kwargs["device"] = device

    results = model.predict(**kwargs)
    if len(results) == 0:
        return []

    result = results[0]
    names = result.names

    detections: List[Dict[str, Any]] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for i, (box, score, cls_id) in enumerate(zip(boxes_xyxy, confs, clss)):
            x1, y1, x2, y2 = box.tolist()
            class_name = names.get(int(cls_id), str(cls_id))

            detections.append(
                {
                    "variant": variant_name,
                    "image_path": str(image_path),
                    "detection_index": int(i),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": float(score),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "area": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
                }
            )

    annotated = result.plot()
    annotated_path = output_dir / f"{safe_name(variant_name)}_yolo_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    if save_crops and result.boxes is not None and len(result.boxes) > 0:
        crop_dir = output_dir / f"{safe_name(variant_name)}_crops"
        ensure_dir(crop_dir)

        img = read_image(image_path)
        H, W = img.shape[:2]

        for det in detections:
            x1 = int(max(0, min(W - 1, round(det["x1"]))))
            y1 = int(max(0, min(H - 1, round(det["y1"]))))
            x2 = int(max(0, min(W, round(det["x2"]))))
            y2 = int(max(0, min(H, round(det["y2"]))))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crop_name = (
                f'{det["detection_index"]:03d}_'
                f'{safe_name(det["class_name"])}_'
                f'{det["confidence"]:.2f}.png'
            )
            cv2.imwrite(str(crop_dir / crop_name), crop)

    print(
        f"[DONE] {variant_name}: {len(detections)} detections | "
        f"annotated={annotated_path}"
    )

    return detections


def write_detection_csv(detections: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "variant",
        "image_path",
        "detection_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "area",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detections:
            writer.writerow(row)


def build_summary(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}

    for det in detections:
        key = (det["variant"], det["class_name"])
        grouped.setdefault(key, []).append(det)

    summary_rows = []

    variants = sorted(set(det["variant"] for det in detections))
    for variant in variants:
        variant_dets = [d for d in detections if d["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "class_name": "__ALL__",
                "count": len(variant_dets),
                "mean_confidence": float(np.mean([d["confidence"] for d in variant_dets])) if variant_dets else 0.0,
                "max_confidence": float(np.max([d["confidence"] for d in variant_dets])) if variant_dets else 0.0,
                "total_box_area": float(np.sum([d["area"] for d in variant_dets])) if variant_dets else 0.0,
            }
        )

    for (variant, class_name), rows in sorted(grouped.items()):
        confs = [r["confidence"] for r in rows]
        areas = [r["area"] for r in rows]
        summary_rows.append(
            {
                "variant": variant,
                "class_name": class_name,
                "count": len(rows),
                "mean_confidence": float(np.mean(confs)),
                "max_confidence": float(np.max(confs)),
                "total_box_area": float(np.sum(areas)),
            }
        )

    return summary_rows


def write_summary_csv(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "variant",
        "class_name",
        "count",
        "mean_confidence",
        "max_confidence",
        "total_box_area",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--original", required=True, help="Original stitched panorama.")
    parser.add_argument("--masked", required=True, help="Masked stitched panorama.")
    parser.add_argument("--inpainted", required=True, help="Inpainted stitched panorama.")

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model path/name, e.g. yolo11n.pt, yolo11s.pt, custom.pt.",
    )

    parser.add_argument("--output_dir", default="stage4_yolo_output")

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1280)

    parser.add_argument(
        "--device",
        default="",
        help="Device for Ultralytics. Examples: cpu, mps, 0. Leave empty for auto.",
    )

    parser.add_argument("--save_crops", action="store_true")

    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Ultralytics is not installed. Install it with:\n"
            "  python -m pip install ultralytics"
        ) from e

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    image_inputs = {
        "original": Path(args.original),
        "masked": Path(args.masked),
        "inpainted": Path(args.inpainted),
    }

    for name, path in image_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} image: {path}")

    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    all_detections: List[Dict[str, Any]] = []

    for variant_name, image_path in image_inputs.items():
        detections = run_yolo_on_image(
            model=model,
            image_path=image_path,
            variant_name=variant_name,
            output_dir=output_dir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_crops=args.save_crops,
        )
        all_detections.extend(detections)

    detections_csv = output_dir / "yolo_detections.csv"
    summary_csv = output_dir / "yolo_summary.csv"
    json_path = output_dir / "yolo_detections.json"

    write_detection_csv(all_detections, detections_csv)
    summary_rows = build_summary(all_detections)
    write_summary_csv(summary_rows, summary_csv)

    with open(json_path, "w") as f:
        json.dump(all_detections, f, indent=2)

    print(f"[DONE] Detection CSV: {detections_csv}")
    print(f"[DONE] Summary CSV:   {summary_csv}")
    print(f"[DONE] Detection JSON:{json_path}")

    print("\n[SUMMARY]")
    for row in summary_rows:
        if row["class_name"] == "__ALL__":
            print(
                f'  {row["variant"]}: {row["count"]} detections, '
                f'mean_conf={row["mean_confidence"]:.3f}'
            )


if __name__ == "__main__":
    main()
