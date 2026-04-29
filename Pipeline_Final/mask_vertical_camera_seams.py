#!/usr/bin/env python3
"""
mask_vertical_camera_seams.py

Stage 2: Mask ONLY vertical seam regions between adjacent camera panels.

This is simpler than the previous seam masker:
  - It uses the camera_id_map.
  - It finds vertical transitions between camera IDs.
  - It ignores horizontal/top/bottom boundaries.
  - It does NOT mask the natural black gaps unless you explicitly ask for it.
  - White mask pixels = remove/inpaint.
  - Black mask pixels = keep.

Inputs:
  --panorama       pipeline panorama image
  --camera_id_map  color camera-id map from the stitching script

Example:
  python mask_vertical_camera_seams.py \
    --panorama stitch_output/1558060923374527_flat360_intrinsic_warp_v5_tight.png \
    --camera_id_map stitch_output/1558060923374527_flat360_intrinsic_warp_v5_tight_camera_id_map.png \
    --output_dir mask_output \
    --mask_percent 0.5 \
    --max_seam_width_px 80 \
    --full_height 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CAMERA_COLOR_TO_ID = {
    (255, 0, 0): 1,      # SIDE_LEFT
    (0, 255, 255): 2,    # FRONT_LEFT
    (0, 255, 0): 3,      # FRONT
    (255, 255, 0): 4,    # FRONT_RIGHT
    (0, 0, 255): 5,      # SIDE_RIGHT
}


def read_image(path: str | Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def color_id_map_to_labels(id_color: np.ndarray) -> np.ndarray:
    if id_color.ndim == 2:
        return id_color.astype(np.uint8)

    if id_color.shape[2] == 4:
        id_color = id_color[:, :, :3]

    labels = np.zeros(id_color.shape[:2], dtype=np.uint8)

    for bgr, idx in CAMERA_COLOR_TO_ID.items():
        b, g, r = bgr
        match = (
            (id_color[:, :, 0] == b)
            & (id_color[:, :, 1] == g)
            & (id_color[:, :, 2] == r)
        )
        labels[match] = idx

    return labels


def detect_vertical_camera_seams(labels: np.ndarray) -> np.ndarray:
    """
    Detect only left/right camera-ID transitions.

    This does NOT detect:
      - top/bottom boundaries
      - valid/invalid image borders
      - black gap boundaries

    A seam is marked only where adjacent horizontal pixels belong to
    different nonzero camera IDs.
    """
    seam = np.zeros(labels.shape, dtype=np.uint8)

    left = labels[:, :-1]
    right = labels[:, 1:]

    diff = (left != right) & (left > 0) & (right > 0)

    seam[:, :-1][diff] = 255
    seam[:, 1:][diff] = 255

    return seam


def make_vertical_band_mask_from_seams(
    seam: np.ndarray,
    width_px: int,
    restrict_to_valid_rows: bool = True,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Expand vertical seam pixels horizontally only.

    Unlike normal dilation, this does not expand upward/downward.
    It only creates vertical seam bands.
    """
    if width_px <= 0:
        return np.zeros_like(seam)

    H, W = seam.shape
    mask = np.zeros_like(seam)

    seam_cols = np.where(seam.max(axis=0) > 0)[0]

    half = width_px

    for x in seam_cols:
        x0 = max(0, x - half)
        x1 = min(W, x + half + 1)

        if restrict_to_valid_rows and valid_mask is not None:
            rows = seam[:, x] > 0
            if rows.any():
                y_idxs = np.where(rows)[0]
                y0 = max(0, y_idxs.min())
                y1 = min(H, y_idxs.max() + 1)
                mask[y0:y1, x0:x1] = 255
        else:
            mask[:, x0:x1] = 255

    return mask


def build_valid_mask(labels: np.ndarray, panorama: np.ndarray) -> np.ndarray:
    valid = labels > 0
    if valid.sum() > 0:
        return valid.astype(np.uint8) * 255

    if panorama.ndim == 3 and panorama.shape[2] == 4:
        return (panorama[:, :, 3] > 0).astype(np.uint8) * 255

    gray = cv2.cvtColor(panorama[:, :, :3], cv2.COLOR_BGR2GRAY)
    return (gray > 2).astype(np.uint8) * 255


def make_masked_panorama(panorama: np.ndarray, mask: np.ndarray, fill_value: int = 0) -> np.ndarray:
    out = panorama.copy()

    if out.ndim == 2:
        out[mask > 0] = fill_value
        return out

    if out.shape[2] == 4:
        out[mask > 0, :3] = fill_value
        out[mask > 0, 3] = 255
    else:
        out[mask > 0] = (fill_value, fill_value, fill_value)

    return out


def make_overlay(panorama: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if panorama.ndim == 2:
        base = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)
    else:
        base = panorama[:, :, :3].copy()

    overlay = base.copy()
    overlay[mask > 0] = (0, 0, 255)

    return cv2.addWeighted(base, 0.70, overlay, 0.30, 0)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--panorama", required=True)
    parser.add_argument("--camera_id_map", required=True)
    parser.add_argument("--output_dir", default="stage2_vertical_mask_output")

    parser.add_argument(
        "--mask_percent",
        type=float,
        default=0.5,
        help="0=no masking, 1=max_seam_width_px.",
    )

    parser.add_argument(
        "--max_seam_width_px",
        type=int,
        default=80,
        help="Half-width of vertical seam band when mask_percent=1.0.",
    )

    parser.add_argument(
        "--full_height",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 masks full image height at seam columns. 0 masks only rows where seam exists.",
    )

    parser.add_argument(
        "--fill_value",
        type=int,
        default=0,
        help="Fill value for masked panorama.",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pano_path = Path(args.panorama)
    id_path = Path(args.camera_id_map)

    panorama = read_image(pano_path)
    id_color = read_image(id_path)

    labels = color_id_map_to_labels(id_color)

    if panorama.shape[:2] != labels.shape[:2]:
        raise ValueError(
            f"Shape mismatch: panorama={panorama.shape[:2]} id_map={labels.shape[:2]}"
        )

    valid_mask = build_valid_mask(labels, panorama)

    mask_percent = float(np.clip(args.mask_percent, 0.0, 1.0))
    seam_width_px = int(round(mask_percent * args.max_seam_width_px))

    raw_vertical_seams = detect_vertical_camera_seams(labels)

    inpaint_mask = make_vertical_band_mask_from_seams(
        raw_vertical_seams,
        width_px=seam_width_px,
        restrict_to_valid_rows=(args.full_height == 0),
        valid_mask=valid_mask,
    )

    masked = make_masked_panorama(panorama, inpaint_mask, fill_value=args.fill_value)
    overlay = make_overlay(panorama, inpaint_mask)

    stem = pano_path.stem

    mask_path = out_dir / f"{stem}_vertical_seam_mask_p{mask_percent:.2f}.png"
    masked_path = out_dir / f"{stem}_vertical_seam_masked_p{mask_percent:.2f}.png"
    overlay_path = out_dir / f"{stem}_vertical_seam_overlay_p{mask_percent:.2f}.png"
    raw_path = out_dir / f"{stem}_raw_vertical_seams.png"

    cv2.imwrite(str(mask_path), inpaint_mask)
    cv2.imwrite(str(masked_path), masked)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(raw_path), raw_vertical_seams)

    print(f"[INFO] mask_percent:       {mask_percent}")
    print(f"[INFO] seam half-width px: {seam_width_px}")
    print(f"[DONE] Inpaint mask:       {mask_path}")
    print(f"[DONE] Masked panorama:    {masked_path}")
    print(f"[DONE] Overlay:            {overlay_path}")
    print(f"[DONE] Raw vertical seams: {raw_path}")


if __name__ == "__main__":
    main()
