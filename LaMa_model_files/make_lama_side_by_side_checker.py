import os
import cv2
import numpy as np
from glob import glob

# --------------------------------------------------
# Paths
# --------------------------------------------------
IMAGE_DIR = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/images"
MASK_DIR = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/masks"
LAMA_DIR = "/mnt/d/edgeconnect_project/lama_outputs_threecam"
OUT_DIR = "/mnt/d/edgeconnect_project/lama_side_by_side_checker"

EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# If True, masked region is white (255) in the mask
HOLE_IS_WHITE = True

# Fill color for masked preview
FILL_VALUE = 0  # 0 = black


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_image_files(image_dir):
    files = []
    for ext in EXTS:
        files.extend(glob(os.path.join(image_dir, ext)))
    return sorted(files)


def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def load_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def make_masked_preview(img, mask, hole_is_white=True, fill_value=0):
    masked = img.copy()
    if hole_is_white:
        hole = mask > 127
    else:
        hole = mask < 127
    masked[hole] = fill_value
    return masked


def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def add_label(img, text):
    out = img.copy()
    cv2.putText(
        out, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, cv2.LINE_AA
    )
    return out


def main():
    ensure_dir(OUT_DIR)

    image_files = collect_image_files(IMAGE_DIR)
    if len(image_files) == 0:
        print(f"No images found in: {IMAGE_DIR}")
        return

    saved = 0
    skipped = 0

    for i, img_path in enumerate(image_files):
        s = stem(img_path)
        mask_path = os.path.join(MASK_DIR, s + ".png")
        lama_path = os.path.join(LAMA_DIR, s + ".png")

        if not os.path.exists(mask_path):
            print(f"[skip] missing mask: {mask_path}")
            skipped += 1
            continue

        if not os.path.exists(lama_path):
            print(f"[skip] missing LaMa result: {lama_path}")
            skipped += 1
            continue

        img = load_bgr(img_path)
        mask = load_mask(mask_path)
        lama = load_bgr(lama_path)

        if img is None:
            print(f"[skip] failed to read image: {img_path}")
            skipped += 1
            continue
        if mask is None:
            print(f"[skip] failed to read mask: {mask_path}")
            skipped += 1
            continue
        if lama is None:
            print(f"[skip] failed to read LaMa result: {lama_path}")
            skipped += 1
            continue

        if img.shape[:2] != mask.shape[:2]:
            print(f"[skip] image/mask size mismatch for {s}: {img.shape[:2]} vs {mask.shape[:2]}")
            skipped += 1
            continue

        if img.shape[:2] != lama.shape[:2]:
            print(f"[skip] image/lama size mismatch for {s}: {img.shape[:2]} vs {lama.shape[:2]}")
            skipped += 1
            continue

        masked = make_masked_preview(img, mask, hole_is_white=HOLE_IS_WHITE, fill_value=FILL_VALUE)
        mask_bgr = mask_to_bgr(mask)

        img_l = add_label(img, "image")
        mask_l = add_label(mask_bgr, "mask")
        masked_l = add_label(masked, "masked")
        lama_l = add_label(lama, "lama")

        panel = cv2.hconcat([img_l, mask_l, masked_l, lama_l])

        out_path = os.path.join(OUT_DIR, s + "_checker.png")
        ok = cv2.imwrite(out_path, panel)

        if not ok:
            print(f"[skip] failed to write: {out_path}")
            skipped += 1
            continue

        saved += 1
        if i < 5:
            print(f"[saved] {out_path}")

    print("\nDone.")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Checker images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()