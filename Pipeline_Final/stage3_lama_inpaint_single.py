#!/usr/bin/env python3
"""
stage3_lama_inpaint_single.py

Stage 3: Inpaint one masked panoramic image using a pretrained LaMa model.

Inputs:
  --image       original / masked panorama image
  --mask        binary inpainting mask
               white/255 = region to inpaint
               black/0   = region to keep

LaMa expected input format:
  sample.png
  sample_mask001.png

This script:
  1. Creates a temporary LaMa input folder
  2. Copies/converts the image and mask into LaMa naming format
  3. Runs LaMa's official bin/predict.py
  4. Copies/renames the output to a clean filename

Example:
  python stage3_lama_inpaint_single.py \
    --image stage2_vertical_mask_output/1558060923374527_flat360_intrinsic_warp_v5_tight_vertical_seam_masked_p0.50.png \
    --mask stage2_vertical_mask_output/1558060923374527_flat360_intrinsic_warp_v5_tight_vertical_seam_mask_p0.50.png \
    --lama_root /mnt/d/lama \
    --model_path /mnt/d/lama/pretrained/big-lama \
    --checkpoint best.ckpt \
    --output_dir stage3_lama_output
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    # Force binary mask: white = inpaint, black = keep.
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def prepare_single_lama_pair(
    image_path: Path,
    mask_path: Path,
    temp_input_dir: Path,
    sample_name: str = "sample",
) -> tuple[Path, Path]:
    if temp_input_dir.exists():
        shutil.rmtree(temp_input_dir)
    ensure_dir(temp_input_dir)

    img = read_bgr(image_path)
    mask = read_mask(mask_path)

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch: image={img.shape[:2]}, mask={mask.shape[:2]}"
        )

    lama_img_path = temp_input_dir / f"{sample_name}.png"
    lama_mask_path = temp_input_dir / f"{sample_name}_mask001.png"

    ok_img = cv2.imwrite(str(lama_img_path), img)
    ok_mask = cv2.imwrite(str(lama_mask_path), mask)

    if not ok_img:
        raise RuntimeError(f"Failed to write LaMa image: {lama_img_path}")
    if not ok_mask:
        raise RuntimeError(f"Failed to write LaMa mask: {lama_mask_path}")

    return lama_img_path, lama_mask_path


def run_lama_predict(
    lama_root: Path,
    model_path: Path,
    checkpoint: str,
    temp_input_dir: Path,
    output_dir: Path,
    gpu: int,
    refine: bool,
) -> None:
    predict_script = lama_root / "bin" / "predict.py"

    if not predict_script.exists():
        raise FileNotFoundError(f"Could not find LaMa predictor: {predict_script}")

    ensure_dir(output_dir)

    env = os.environ.copy()

    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(lama_root) if old_pythonpath == "" else str(lama_root) + ":" + old_pythonpath

    if gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "python",
        str(predict_script),
        f"model.path={model_path}",
        f"model.checkpoint={checkpoint}",
        f"indir={temp_input_dir}",
        f"outdir={output_dir}",
    ]

    if refine:
        cmd.append("refine=True")

    print("[INFO] Running LaMa command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, cwd=str(lama_root), env=env)


def find_lama_output(output_dir: Path, sample_name: str = "sample") -> Path:
    candidates = [
        output_dir / f"{sample_name}.png",
        output_dir / f"{sample_name}_mask001.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    pngs = sorted(output_dir.glob("*.png"))
    if len(pngs) == 0:
        raise RuntimeError(f"No PNG output found in: {output_dir}")

    return pngs[0]


def make_checker(
    image_path: Path,
    mask_path: Path,
    inpaint_path: Path,
    output_path: Path,
) -> None:
    img = read_bgr(image_path)
    mask = read_mask(mask_path)
    inp = read_bgr(inpaint_path)

    if inp.shape[:2] != img.shape[:2]:
        inp = cv2.resize(inp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    masked = img.copy()
    masked[mask > 0] = (0, 0, 0)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def label(im: np.ndarray, text: str) -> np.ndarray:
        out = im.copy()
        cv2.putText(
            out,
            text,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return out

    panel = cv2.hconcat([
        label(img, "input"),
        label(mask_bgr, "mask"),
        label(masked, "masked"),
        label(inp, "inpainted"),
    ])

    cv2.imwrite(str(output_path), panel)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, help="Input panorama image.")
    parser.add_argument("--mask", required=True, help="Binary mask. White=hole/inpaint.")

    parser.add_argument("--lama_root", default="/mnt/d/lama")
    parser.add_argument("--model_path", default="/mnt/d/lama/pretrained/big-lama")
    parser.add_argument("--checkpoint", default="best.ckpt")

    parser.add_argument("--output_dir", default="stage3_lama_output")
    parser.add_argument("--temp_input_dir", default="stage3_lama_temp_input")

    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use -1 for CPU/no CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--refine", action="store_true")

    parser.add_argument(
        "--sample_name",
        default="sample",
        help="Internal filename used for LaMa input.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)

    lama_root = Path(args.lama_root).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    temp_input_dir = Path(args.temp_input_dir).resolve()

    ensure_dir(output_dir)

    print("[INFO] Preparing LaMa input pair")
    lama_img, lama_mask = prepare_single_lama_pair(
        image_path=image_path,
        mask_path=mask_path,
        temp_input_dir=temp_input_dir,
        sample_name=args.sample_name,
    )

    print(f"[INFO] LaMa image: {lama_img}")
    print(f"[INFO] LaMa mask:  {lama_mask}")

    raw_lama_out_dir = output_dir / "raw_lama"
    if raw_lama_out_dir.exists():
        shutil.rmtree(raw_lama_out_dir)
    ensure_dir(raw_lama_out_dir)

    run_lama_predict(
        lama_root=lama_root,
        model_path=model_path,
        checkpoint=args.checkpoint,
        temp_input_dir=temp_input_dir,
        output_dir=raw_lama_out_dir,
        gpu=args.gpu,
        refine=args.refine,
    )

    raw_output = find_lama_output(raw_lama_out_dir, sample_name=args.sample_name)

    stem = image_path.stem
    final_path = output_dir / f"{stem}_lama_inpainted.png"
    checker_path = output_dir / f"{stem}_lama_checker.png"

    shutil.copy2(raw_output, final_path)

    make_checker(
        image_path=image_path,
        mask_path=mask_path,
        inpaint_path=final_path,
        output_path=checker_path,
    )

    print(f"[DONE] Inpainted image: {final_path}")
    print(f"[DONE] Checker image:   {checker_path}")


if __name__ == "__main__":
    main()
