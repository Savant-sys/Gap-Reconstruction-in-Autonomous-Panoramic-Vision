#!/usr/bin/env python3
"""
waymo_flat360_intrinsic_warp_v5_tight.py

Pipeline-ready tight-cropped sparse 360 panorama with natural camera gaps for Waymo segmented cameras.

This version improves over the yaw-layout panel version:
  - It still places cameras by extrinsic yaw.
  - It preserves natural black gaps between views.
  - It does NOT force/crop gaps using coverage_scale.
  - Each camera image is warped using intrinsics:
        pixel -> undistorted camera ray -> local yaw/pitch -> flat angular panel

This produces a more camera-correct affine/perspective-like warp, especially for side views.

Requires:
  waymo_loader.py in same folder.
  
Example:
/usr/local/bin/python3 waymo_flat360_intrinsic_warp_v5_tight.py \
  --camera_image_parquet /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/Dataset/dataset/camera_image/10504764403039842352_460_000_480_000.parquet \
  --camera_calibration_parquet /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/Dataset/dataset/camera_calibration/10504764403039842352_460_000_480_000.parquet \
  --frame_index 0 \
  --output_dir stitch_output \
  --panorama_width 4800 \
  --panorama_height 900 \
  --vertical_fov_deg 55 \
  --horizon 0.52 \
  --sample_step 2 \
  --splat_radius 1 \
  --crop_to_content \
  --crop_pad_x 0 \
  --crop_pad_y 0 \
  --save_rgba
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from waymo_loader import (
    load_camera_images_from_parquet,
    load_camera_calibration,
    list_frames_in_parquet,
)


ORDER = ["SIDE_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT"]

CAMERA_ID = {
    "SIDE_LEFT": 1,
    "FRONT_LEFT": 2,
    "FRONT": 3,
    "FRONT_RIGHT": 4,
    "SIDE_RIGHT": 5,
}

CAMERA_COLOR = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (0, 0, 255),
}


def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0


def camera_forward_yaw(calib: dict) -> float:
    R_v2c = calib["R"]
    R_c2v = R_v2c.T

    # Waymo camera optical forward is +X.
    fwd_vehicle = R_c2v @ np.array([1.0, 0.0, 0.0])
    fwd_vehicle = fwd_vehicle / (np.linalg.norm(fwd_vehicle) + 1e-9)

    return float(np.degrees(np.arctan2(fwd_vehicle[1], fwd_vehicle[0])))


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    s = target_h / h
    return cv2.resize(img, (int(round(w * s)), target_h), interpolation=cv2.INTER_AREA)


def save_camera_strip(images, out_path):
    pieces = []
    for cam in ORDER:
        if cam not in images:
            continue
        im = resize_to_height(images[cam], 320)
        cv2.putText(im, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        pieces.append(im)
    if pieces:
        cv2.imwrite(str(out_path), np.concatenate(pieces, axis=1))


def estimate_horizontal_fov_deg(calib, image_shape, border_crop_frac=0.04):
    h, w = image_shape[:2]
    K = calib["K"].astype(np.float64)
    D = calib["D"].astype(np.float64)

    x0 = w * border_crop_frac
    x1 = w * (1.0 - border_crop_frac)
    y = h * 0.5

    pts = np.array([[[x0, y]], [[x1, y]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, D).reshape(-1, 2)

    yaw0 = np.degrees(np.arctan2(und[0, 0], 1.0))
    yaw1 = np.degrees(np.arctan2(und[1, 0], 1.0))

    return float(abs(yaw1 - yaw0))


def colorize_id_map(id_map):
    out = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    for idx, color in CAMERA_COLOR.items():
        out[id_map == idx] = color
    return out


def yaw_to_x(yaw_deg, W, yaw_center_deg):
    """
    Display convention:
      FRONT yaw 0 is center.
      Positive yaw is vehicle-left, therefore appears left of center.
    """
    display_yaw = wrap_deg(yaw_deg - yaw_center_deg)
    return (0.5 - display_yaw / 360.0) * (W - 1)


def pitch_to_y(pitch_deg, H, vertical_fov_deg, horizon):
    max_tan = np.tan(np.deg2rad(vertical_fov_deg) / 2.0)
    z = np.tan(np.deg2rad(pitch_deg))
    return horizon * H - (z / max_tan) * (H / 2.0)


def splat(canvas_sum, weight_sum, id_map, x, y, colors, cam_id, radius):
    H, W = weight_sum.shape

    xi = np.round(x).astype(np.int32)
    yi = np.round(y).astype(np.int32)

    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi = xi[valid]
    yi = yi[valid]
    colors = colors[valid].astype(np.float32)

    count = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx = xi + dx
            yy = yi + dy

            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            if not np.any(ok):
                continue

            canvas_sum[yy[ok], xx[ok]] += colors[ok]
            weight_sum[yy[ok], xx[ok]] += 1.0
            id_map[yy[ok], xx[ok]] = cam_id
            count += int(ok.sum())

    return count


def project_camera_intrinsic_warp(
    img,
    calib,
    cam_name,
    cam_yaw,
    canvas_sum,
    weight_sum,
    id_map,
    pano_w,
    pano_h,
    yaw_center_deg,
    vertical_fov_deg,
    horizon,
    keep_half_fov_deg,
    sample_step,
    splat_radius,
    border_crop_frac,
):
    """
    Forward warp source pixels into the final flat 360 canvas.

    Key difference from simple panel paste:
      source pixel -> undistorted normalized coords -> local yaw/pitch
      local yaw is added to extrinsic camera yaw.
    """
    h, w = img.shape[:2]

    bx = int(round(w * border_crop_frac))
    by = int(round(h * border_crop_frac))

    xs = np.arange(bx, w - bx, sample_step, dtype=np.float32)
    ys = np.arange(by, h - by, sample_step, dtype=np.float32)

    uu, vv = np.meshgrid(xs, ys)
    pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(np.float32)
    pts_cv = pts.reshape(-1, 1, 2)

    K = calib["K"].astype(np.float64)
    D = calib["D"].astype(np.float64)

    und = cv2.undistortPoints(pts_cv, K, D).reshape(-1, 2)

    x_norm = und[:, 0]
    y_norm = und[:, 1]

    # Waymo camera optical axis is +X.
    # Local angular coordinates in camera frame.
    local_yaw = np.degrees(np.arctan2(x_norm, 1.0))

    # image y is down; convert to pitch up.
    local_pitch = np.degrees(
        np.arctan2(
            -y_norm,
            np.sqrt(1.0 + x_norm * x_norm),
        )
    )

    # Keep only central angular sector so real gaps remain visible.
    angular_valid = np.abs(local_yaw) <= keep_half_fov_deg

    # Correct Waymo image horizontal direction for display: image-right corresponds to lower vehicle yaw.
    global_yaw = wrap_deg(cam_yaw - local_yaw)

    x = yaw_to_x(global_yaw, pano_w, yaw_center_deg)
    y = pitch_to_y(local_pitch, pano_h, vertical_fov_deg, horizon)

    px = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    colors = img[py, px]

    x = x[angular_valid]
    y = y[angular_valid]
    colors = colors[angular_valid]

    count = splat(
        canvas_sum,
        weight_sum,
        id_map,
        x,
        y,
        colors,
        CAMERA_ID[cam_name],
        splat_radius,
    )

    return int(angular_valid.sum()), count


def draw_debug_yaw(canvas, yaws, yaw_center_deg):
    out = canvas.copy()
    H, W = out.shape[:2]

    for cam, yaw in yaws.items():
        x = int(round(yaw_to_x(yaw, W, yaw_center_deg))) % W
        cv2.line(out, (x, 0), (x, H - 1), (255, 255, 255), 2)
        cv2.putText(
            out,
            f"{cam} yaw={yaw:.1f}",
            (max(5, min(W - 270, x - 130)), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out



def crop_to_valid_content(canvas, coverage, id_map, pad_x=20, pad_y=20):
    """
    Crop away empty black padding while preserving all projected camera pixels.

    Uses coverage mask, not pixel intensity, so true dark image pixels are preserved.
    """
    ys, xs = np.where(coverage > 0)

    if len(xs) == 0 or len(ys) == 0:
        return canvas, coverage, id_map

    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(canvas.shape[1], int(xs.max()) + pad_x + 1)

    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(canvas.shape[0], int(ys.max()) + pad_y + 1)

    return (
        canvas[y0:y1, x0:x1],
        coverage[y0:y1, x0:x1],
        id_map[y0:y1, x0:x1],
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera_image_parquet", required=True)
    parser.add_argument("--camera_calibration_parquet", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output_dir", default="stitch_output")

    parser.add_argument("--panorama_width", type=int, default=4800)
    parser.add_argument("--panorama_height", type=int, default=900)

    parser.add_argument("--vertical_fov_deg", type=float, default=55.0)
    parser.add_argument("--horizon", type=float, default=0.52)
    parser.add_argument("--yaw_center_deg", type=float, default=0.0)

    parser.add_argument(
        "--coverage_scale",
        type=float,
        default=1.0,
        help="Deprecated in v3. Kept only so older commands do not crash.",
    )
    parser.add_argument("--min_half_width_deg", type=float, default=0.0)
    parser.add_argument("--max_half_width_deg", type=float, default=180.0)

    parser.add_argument("--sample_step", type=int, default=2)
    parser.add_argument("--splat_radius", type=int, default=1)
    parser.add_argument("--border_crop_frac", type=float, default=0.035)

    parser.add_argument("--draw_debug_yaw", action="store_true")

    parser.add_argument(
        "--crop_to_content",
        action="store_true",
        help="Crop output to all valid projected pixels while preserving all image data.",
    )
    parser.add_argument("--crop_pad_x", type=int, default=0)
    parser.add_argument("--crop_pad_y", type=int, default=0)

    parser.add_argument(
        "--save_rgba",
        action="store_true",
        help="Also save an RGBA PNG where invalid black background is transparent.",
    )

    parser.add_argument(
        "--fill_invalid",
        type=int,
        default=-1,
        help="If >=0, fill invalid background pixels with this grayscale value, e.g. 128.",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = list_frames_in_parquet(args.camera_image_parquet)
    ts = timestamps[args.frame_index]

    images = load_camera_images_from_parquet(
        args.camera_image_parquet,
        frame_timestamp_micros=ts,
    )
    calibs = load_camera_calibration(args.camera_calibration_parquet)

    save_camera_strip(images, out_dir / f"{ts}_flat360_intrinsic_warp_camera_strip.png")

    H = args.panorama_height
    W = args.panorama_width

    canvas_sum = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    id_map = np.zeros((H, W), dtype=np.uint8)

    yaws = {}

    print("[INFO] Camera yaw order and angular coverage:")

    for cam in ORDER:
        if cam not in images or cam not in calibs:
            print(f"  {cam}: missing")
            continue

        yaw = camera_forward_yaw(calibs[cam])
        yaws[cam] = yaw

        fov = estimate_horizontal_fov_deg(
            calibs[cam],
            images[cam].shape,
            border_crop_frac=args.border_crop_frac,
        )

        keep_half = 180.0

        print(
            f"  {cam:12s} yaw={yaw:8.2f} deg, "
            f"estimated_hfov={fov:6.2f}, using FULL intrinsic coverage"
        )

        kept, splatted = project_camera_intrinsic_warp(
            img=images[cam],
            calib=calibs[cam],
            cam_name=cam,
            cam_yaw=yaw,
            canvas_sum=canvas_sum,
            weight_sum=weight_sum,
            id_map=id_map,
            pano_w=W,
            pano_h=H,
            yaw_center_deg=args.yaw_center_deg,
            vertical_fov_deg=args.vertical_fov_deg,
            horizon=args.horizon,
            keep_half_fov_deg=keep_half,
            sample_step=args.sample_step,
            splat_radius=args.splat_radius,
            border_crop_frac=args.border_crop_frac,
        )

        print(f"      kept source samples={kept}, splatted pixels={splatted}")

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    valid = weight_sum > 0
    canvas[valid] = np.clip(canvas_sum[valid] / weight_sum[valid, None], 0, 255).astype(np.uint8)

    coverage = (weight_sum > 0).astype(np.uint8) * 255

    if args.draw_debug_yaw:
        debug = draw_debug_yaw(canvas, yaws, args.yaw_center_deg)
        debug_path = out_dir / f"{ts}_flat360_intrinsic_warp_v4_pipeline_debug_yaw.png"
        cv2.imwrite(str(debug_path), debug)
        print(f"[DONE] Debug yaw:     {debug_path}")

    if args.crop_to_content:
        canvas, coverage, id_map = crop_to_valid_content(
            canvas,
            coverage,
            id_map,
            pad_x=args.crop_pad_x,
            pad_y=args.crop_pad_y,
        )

    # Optional: make invalid background non-black for easier inspection.
    canvas_to_save = canvas.copy()
    if args.fill_invalid >= 0:
        invalid = coverage == 0
        v = int(np.clip(args.fill_invalid, 0, 255))
        canvas_to_save[invalid] = (v, v, v)

    id_color = colorize_id_map(id_map)

    pano_path = out_dir / f"{ts}_flat360_intrinsic_warp_v5_tight.png"
    coverage_path = out_dir / f"{ts}_flat360_intrinsic_warp_v5_tight_coverage_mask.png"
    id_path = out_dir / f"{ts}_flat360_intrinsic_warp_v5_tight_camera_id_map.png"

    cv2.imwrite(str(pano_path), canvas_to_save)
    cv2.imwrite(str(coverage_path), coverage)
    cv2.imwrite(str(id_path), id_color)

    print(f"[DONE] Tight panorama: {pano_path}")
    print(f"[DONE] Coverage mask:  {coverage_path}")
    print(f"[DONE] Camera ID map:  {id_path}")
    print(f"[INFO] Output shape:   {canvas_to_save.shape[1]}x{canvas_to_save.shape[0]}")

    if args.save_rgba:
        rgba = np.dstack([canvas, coverage])
        rgba_path = out_dir / f"{ts}_flat360_intrinsic_warp_v5_tight_rgba.png"
        cv2.imwrite(str(rgba_path), rgba)
        print(f"[DONE] RGBA panorama:  {rgba_path}")


if __name__ == "__main__":
    main()
