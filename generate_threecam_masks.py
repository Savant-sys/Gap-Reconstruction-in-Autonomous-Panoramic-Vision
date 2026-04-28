import os
import cv2
import numpy as np
from glob import glob

# --------------------------------------------------
# Paths (adjust if needed)
# --------------------------------------------------
IMAGE_DIR = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/images"
MASK_DIR  = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/masks"

# --------------------------------------------------
# Settings
# --------------------------------------------------
# Width of seam mask (increase for wider gap)
SEAM_WIDTH_RATIO = 0.12   # try 0.05 ~ 0.12

# Vertical coverage (1.0 = full height)
MASK_HEIGHT_RATIO = 1.0

CENTER_VERTICALLY = True

# Seam detection search window
SEARCH_WINDOW_RATIO = 0.15

# Smoothing
SMOOTH_KERNEL = 31

# Mask values
BACKGROUND_VALUE = 0
MASK_VALUE = 255

EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


def get_image_files(image_dir):
    files = []
    for ext in EXTS:
        files.extend(glob(os.path.join(image_dir, ext)))
    return sorted(files)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def smooth_1d(x, ksize=31):
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones(ksize, dtype=np.float32) / ksize
    return np.convolve(x, kernel, mode="same")


def detect_seams(img):
    """
    Detect two seam positions based on vertical difference energy
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray[:, 1:] - gray[:, :-1])
    score = diff.mean(axis=0)

    score = smooth_1d(score, SMOOTH_KERNEL)

    h, w = gray.shape
    left_expected = w / 3.0
    right_expected = 2 * w / 3.0
    win = int(w * SEARCH_WINDOW_RATIO)

    # left seam
    l1 = max(0, int(left_expected - win))
    l2 = min(w - 1, int(left_expected + win))
    left_x = np.argmax(score[l1:l2]) + l1

    # right seam
    r1 = max(0, int(right_expected - win))
    r2 = min(w - 1, int(right_expected + win))
    right_x = np.argmax(score[r1:r2]) + r1

    return int(left_x), int(right_x)


def create_mask(h, w, seam_positions):
    mask = np.full((h, w), BACKGROUND_VALUE, dtype=np.uint8)

    seam_width = max(1, int(w * SEAM_WIDTH_RATIO))

    rect_h = int(h * MASK_HEIGHT_RATIO)
    rect_h = min(rect_h, h)

    if CENTER_VERTICALLY:
        y1 = (h - rect_h) // 2
    else:
        y1 = 0
    y2 = y1 + rect_h

    for cx in seam_positions:
        x1 = max(0, cx - seam_width // 2)
        x2 = min(w, x1 + seam_width)
        mask[y1:y2, x1:x2] = MASK_VALUE

    return mask


def main():
    os.makedirs(MASK_DIR, exist_ok=True)

    image_files = get_image_files(IMAGE_DIR)
    print(f"Found {len(image_files)} images.")

    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] {img_path}")
            continue

        h, w = img.shape[:2]

        left_x, right_x = detect_seams(img)
        mask = create_mask(h, w, [left_x, right_x])

        out_path = os.path.join(MASK_DIR, stem(img_path) + ".png")
        cv2.imwrite(out_path, mask)

        if i < 5:
            print(f"[saved] {out_path}")
            print(f"        seams: {left_x}, {right_x}")

    print("\nDone.")
    print(f"Masks saved to: {MASK_DIR}")


if __name__ == "__main__":
    main()