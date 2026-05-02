import os
import sys
import csv
import json
import shutil
import subprocess
from glob import glob
from typing import List, Optional, Dict, Any

import cv2
import torch
from ultralytics import YOLO


# =========================================================
# USER SETTINGS
# =========================================================

# LaMa
LAMA_ROOT = "/mnt/d/lama"
LAMA_MODEL_PATH = "/mnt/d/lama/pretrained/big-lama"
LAMA_CHECKPOINT = "best.ckpt"

# Input dataset
IMAGE_DIR = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/images"
MASK_DIR = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/masks_narrow"

# Temporary LaMa-format folder
TEMP_INPUT_DIR = "/mnt/d/lama/tmp_waymo_lama_input_yolo11"

# Outputs
LAMA_OUT_DIR = "/mnt/d/edgeconnect_project/lama_outputs_threecam_yolo11"
YOLO_OUT_DIR = "/mnt/d/edgeconnect_project/lama_yolo11_outputs"

# YOLO11
YOLO_MODEL = "yolo11s.pt"   # try yolo11n.pt for faster inference
DEVICE = 0                  # GPU index, or "cpu"
IMGSZ = 1280
CONF_THRES = 0.25
IOU_THRES = 0.45

# Optional class filtering.
# Set to None to keep all classes.
# Example for road scenes:
KEEP_CLASSES: Optional[List[str]] = ["car", "truck", "bus", "person", "motorcycle", "bicycle"]

# If True, keep only detections whose box center lies inside / near the seam mask
USE_MASK_REGION_FILTER = False

# Dilate seam mask before region filter
MASK_DILATE_KERNEL = 31

# Output files
CSV_PATH = os.path.join(YOLO_OUT_DIR, "detections.csv")
JSON_PATH = os.path.join(YOLO_OUT_DIR, "detections.json")


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def collect_image_files(folder: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files)


def prepare_lama_input(images_dir: str, masks_dir: str, temp_input_dir: str) -> int:
    """
    LaMa expects files like:
      000001.png
      000001_mask001.png
    in the same folder.
    """
    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    ensure_dir(temp_input_dir)

    image_files = collect_image_files(images_dir)
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in: {images_dir}")

    copied = 0
    skipped = 0

    for img_path in image_files:
        s = stem(img_path)
        mask_path = os.path.join(masks_dir, s + ".png")

        if not os.path.exists(mask_path):
            print(f"[skip] missing mask: {mask_path}")
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[skip] failed to read image: {img_path}")
            skipped += 1
            continue

        if mask is None:
            print(f"[skip] failed to read mask: {mask_path}")
            skipped += 1
            continue

        dst_img = os.path.join(temp_input_dir, s + ".png")
        dst_mask = os.path.join(temp_input_dir, s + "_mask001.png")

        ok1 = cv2.imwrite(dst_img, img)
        ok2 = cv2.imwrite(dst_mask, mask)

        if not ok1 or not ok2:
            print(f"[skip] failed to write LaMa pair for {s}")
            skipped += 1
            continue

        copied += 1

    print(f"Prepared LaMa input pairs: copied={copied}, skipped={skipped}")
    if copied == 0:
        raise RuntimeError("No valid LaMa input pairs were created.")
    return copied


def run_lama() -> None:
    """
    Runs official LaMa predict.py.
    Assumes your predict.py is already patched to use CUDA when available.
    """
    ensure_dir(LAMA_OUT_DIR)

    predict_script = os.path.join(LAMA_ROOT, "bin", "predict.py")
    if not os.path.exists(predict_script):
        raise FileNotFoundError(f"Missing LaMa predict.py: {predict_script}")

    config_path = os.path.join(LAMA_MODEL_PATH, "config.yaml")
    ckpt_path = os.path.join(LAMA_MODEL_PATH, "models", LAMA_CHECKPOINT)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing LaMa config: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing LaMa checkpoint: {ckpt_path}")

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = LAMA_ROOT if old_pythonpath == "" else LAMA_ROOT + ":" + old_pythonpath
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if isinstance(DEVICE, int):
        env["CUDA_VISIBLE_DEVICES"] = str(DEVICE)

    cmd = [
        sys.executable,
        predict_script,
        f"model.path={LAMA_MODEL_PATH}",
        f"model.checkpoint={LAMA_CHECKPOINT}",
        f"indir={TEMP_INPUT_DIR}",
        f"outdir={LAMA_OUT_DIR}",
    ]

    print("\nRunning LaMa...")
    result = subprocess.run(
        cmd,
        cwd=LAMA_ROOT,
        env=env,
        text=True,
        capture_output=True
    )

    print("===== LaMa stdout =====")
    print(result.stdout)
    print("===== LaMa stderr =====")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("LaMa inference failed.")


def load_mask_for_filter(mask_dir: str, image_stem: str) -> Optional[Any]:
    mask_path = os.path.join(mask_dir, image_stem + ".png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    if MASK_DILATE_KERNEL > 1:
        k = np.ones((MASK_DILATE_KERNEL, MASK_DILATE_KERNEL), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


def keep_detection_by_class(class_name: str) -> bool:
    if KEEP_CLASSES is None:
        return True
    return class_name in set(KEEP_CLASSES)


def keep_detection_by_mask(box_xyxy: List[float], mask) -> bool:
    """
    Keep a detection if its center lies in the masked region.
    """
    if mask is None:
        return True

    x1, y1, x2, y2 = box_xyxy
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))

    h, w = mask.shape[:2]
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    return mask[cy, cx] > 127


def draw_boxes(image, detections: List[Dict[str, Any]]):
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
    return out


def run_yolo11() -> None:
    ensure_dir(YOLO_OUT_DIR)

    model = YOLO(YOLO_MODEL)

    lama_images = collect_image_files(LAMA_OUT_DIR)
    if len(lama_images) == 0:
        raise RuntimeError(f"No LaMa outputs found in: {LAMA_OUT_DIR}")

    all_json = []

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        for img_path in lama_images:
            s = stem(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[skip] failed to read LaMa output: {img_path}")
                continue

            seam_mask = load_mask_for_filter(MASK_DIR, s) if USE_MASK_REGION_FILTER else None

            results = model.predict(
                source=img_path,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )

            if len(results) == 0:
                continue

            result = results[0]
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    class_name = result.names[cls_id]
                    confidence = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()

                    if not keep_detection_by_class(class_name):
                        continue

                    if USE_MASK_REGION_FILTER and not keep_detection_by_mask(xyxy, seam_mask):
                        continue

                    det = {
                        "image": os.path.basename(img_path),
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "xyxy": xyxy,
                    }
                    detections.append(det)

                    x1, y1, x2, y2 = xyxy
                    writer.writerow([
                        det["image"], cls_id, class_name, confidence, x1, y1, x2, y2
                    ])

            all_json.append({
                "image": os.path.basename(img_path),
                "detections": detections
            })

            annotated = draw_boxes(img, detections)
            out_path = os.path.join(YOLO_OUT_DIR, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)

    with open(JSON_PATH, "w") as f:
        json.dump(all_json, f, indent=2)

    print(f"\nYOLO11 annotated outputs saved to: {YOLO_OUT_DIR}")
    print(f"CSV detections saved to: {CSV_PATH}")
    print(f"JSON detections saved to: {JSON_PATH}")


def main():
    print("=== Device Check ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    prepare_lama_input(IMAGE_DIR, MASK_DIR, TEMP_INPUT_DIR)
    run_lama()
    run_yolo11()
    print("\nLaMa + YOLO11 pipeline completed.")


if __name__ == "__main__":
    import numpy as np
    main()