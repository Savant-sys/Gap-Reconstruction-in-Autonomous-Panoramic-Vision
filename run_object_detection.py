import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(
        description="Run YOLO object detection on a panorama output image."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input image, e.g. test_output.png",
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help='YOLO model name or path, e.g. "yolo11n.pt" or "path/to/best.pt"',
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_output.png"),
        help="Annotated output image path.",
    )
    ap.add_argument(
        "--json_output",
        type=Path,
        default=Path("yolo_output.json"),
        help="JSON file for detections.",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size.",
    )

    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    image = cv2.imread(str(args.input))
    if image is None:
        raise ValueError(f"Could not read image: {args.input}")

    model = YOLO(args.model)

    results = model(
        str(args.input),
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=True,
    )

    result = results[0]

    # Save annotated image
    annotated = result.plot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)

    # Save JSON detections
    detections = []
    names = result.names

    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes_xyxy, confs, clses):
            x1, y1, x2, y2 = box.tolist()
            cls_id = int(cls_id)
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": float(conf),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_image": str(args.input),
                "model": args.model,
                "num_detections": len(detections),
                "detections": detections,
            },
            f,
            indent=2,
        )

    print(f"Saved annotated image to: {args.output}")
    print(f"Saved detection JSON to: {args.json_output}")
    print(f"Detections: {len(detections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())