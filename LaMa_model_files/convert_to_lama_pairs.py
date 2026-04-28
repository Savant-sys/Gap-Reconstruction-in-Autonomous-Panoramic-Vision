import os
import cv2
import shutil
from glob import glob

SRC_ROOT = "/mnt/d/waymo_data/three_camera_warped_torch_gpu_split"
DST_ROOT = "/mnt/d/waymo_data/lama_threecam_pairs"

EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_image_files(image_dir):
    files = []
    for ext in EXTS:
        files.extend(glob(os.path.join(image_dir, ext)))
    return sorted(files)


def convert_split(split):
    src_img_dir = os.path.join(SRC_ROOT, split, "images")
    src_mask_dir = os.path.join(SRC_ROOT, split, "masks")
    dst_dir = os.path.join(DST_ROOT, split)

    ensure_dir(dst_dir)

    image_files = collect_image_files(src_img_dir)
    if len(image_files) == 0:
        print(f"[warn] no images found in {src_img_dir}")
        return

    copied = 0
    skipped = 0

    for img_path in image_files:
        s = stem(img_path)
        mask_path = os.path.join(src_mask_dir, s + ".png")

        if not os.path.exists(mask_path):
            print(f"[skip] missing mask for {s}: {mask_path}")
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] failed to read image: {img_path}")
            skipped += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[skip] failed to read mask: {mask_path}")
            skipped += 1
            continue

        dst_img = os.path.join(dst_dir, s + ".png")
        dst_mask = os.path.join(dst_dir, s + "_mask001.png")

        ok1 = cv2.imwrite(dst_img, img)
        ok2 = cv2.imwrite(dst_mask, mask)

        if not ok1 or not ok2:
            print(f"[skip] failed to write pair for {s}")
            skipped += 1
            continue

        copied += 1

    print(f"{split}: copied={copied}, skipped={skipped}, out={dst_dir}")


def main():
    ensure_dir(DST_ROOT)
    convert_split("train")
    convert_split("val")
    print(f"\nDone. LaMa pairs written to: {DST_ROOT}")


if __name__ == "__main__":
    main()