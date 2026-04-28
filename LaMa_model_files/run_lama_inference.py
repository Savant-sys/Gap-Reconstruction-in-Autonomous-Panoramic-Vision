import os
import shutil
import subprocess
import argparse
from glob import glob

import cv2
import torch


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_image_files(image_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(image_dir, ext)))
    return sorted(files)


def prepare_lama_input(images_dir, masks_dir, temp_input_dir):
    """
    LaMa expects:
      image.png
      image_mask001.png

    So we always convert/copy images to PNG.
    """
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
            print(f"[skip] missing mask for {s}: {mask_path}")
            skipped += 1
            continue

        dst_img = os.path.join(temp_input_dir, s + ".png")
        dst_mask = os.path.join(temp_input_dir, s + "_mask001.png")

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

        ok1 = cv2.imwrite(dst_img, img)
        ok2 = cv2.imwrite(dst_mask, mask)

        if not ok1 or not ok2:
            print(f"[skip] failed to write pair for {s}")
            skipped += 1
            continue

        copied += 1

    print(f"Prepared LaMa input folder: {temp_input_dir}")
    print(f"Copied pairs: {copied}, skipped: {skipped}")

    if copied == 0:
        raise RuntimeError("No valid image-mask pairs were copied.")


def main():
    ap = argparse.ArgumentParser()

    # LaMa repo root
    ap.add_argument(
        "--lama_root",
        default="/mnt/d/lama",
        help="Path to cloned LaMa repository"
    )

    # Pretrained model directory
    ap.add_argument(
        "--model_path",
        default="/mnt/d/lama/pretrained/big-lama",
        help="Directory containing config.yaml and models/<checkpoint>"
    )

    ap.add_argument(
        "--checkpoint",
        default="best.ckpt",
        help="Checkpoint file inside model_path/models/"
    )

    # Dataset
    ap.add_argument(
        "--images_dir",
        default="/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/images"
    )
    ap.add_argument(
        "--masks_dir",
        default="/mnt/d/waymo_data/three_camera_warped_torch_gpu_split/val/masks"
    )

    # Temporary LaMa input directory
    ap.add_argument(
        "--temp_input_dir",
        default="/mnt/d/lama/tmp_waymo_lama_input"
    )

    # Output directory
    ap.add_argument(
        "--outdir",
        default="/mnt/d/edgeconnect_project/lama_outputs_threecam"
    )

    # GPU selection
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use"
    )

    # Optional refinement
    ap.add_argument(
        "--refine",
        action="store_true",
        help="Enable LaMa refinement mode"
    )

    args = ap.parse_args()

    lama_root = os.path.abspath(args.lama_root)
    model_path = os.path.abspath(args.model_path)
    images_dir = os.path.abspath(args.images_dir)
    masks_dir = os.path.abspath(args.masks_dir)
    temp_input_dir = os.path.abspath(args.temp_input_dir)
    outdir = os.path.abspath(args.outdir)

    ensure_dir(outdir)

    # --------------------------------------------------
    # Check CUDA
    # --------------------------------------------------
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Using GPU index:", args.gpu)
        print("GPU name:", torch.cuda.get_device_name(args.gpu))
    else:
        print("WARNING: CUDA is not available. LaMa will run on CPU.")

    # --------------------------------------------------
    # Rebuild temp input folder
    # --------------------------------------------------
    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    ensure_dir(temp_input_dir)

    prepare_lama_input(images_dir, masks_dir, temp_input_dir)

    # --------------------------------------------------
    # Launch official LaMa predictor
    # --------------------------------------------------
    predict_script = os.path.join(lama_root, "bin", "predict.py")
    if not os.path.exists(predict_script):
        raise FileNotFoundError(f"Could not find LaMa predictor: {predict_script}")

    env = os.environ.copy()

    # Make sure LaMa repo is importable
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = lama_root if old_pythonpath == "" else lama_root + ":" + old_pythonpath

    # Force selected GPU for the LaMa subprocess
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Optional: reduce CUDA fragmentation issues
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "python",
        predict_script,
        f"model.path={model_path}",
        f"model.checkpoint={args.checkpoint}",
        f"indir={temp_input_dir}",
        f"outdir={outdir}",
    ]

    if args.refine:
        cmd.append("refine=True")

    print("\nRunning LaMa inference command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True, cwd=lama_root, env=env)

    print("\nDone.")
    print(f"LaMa outputs saved to: {outdir}")


if __name__ == "__main__":
    main()