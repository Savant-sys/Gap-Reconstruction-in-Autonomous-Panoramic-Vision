"""
Calibration-based cylindrical panorama from Waymo multi-camera images.

Uses camera intrinsics and extrinsics to project each image onto a common
cylinder (vehicle frame), then composites into one panorama. Avoids the
distortion and black gaps of homography-only stitching.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from waymo_loader import DEFAULT_PANORAMA_ORDER


def _vehicle_to_camera_point(R: np.ndarray, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """P: (3,) or (3,N) in vehicle frame. R,t are vehicle-to-camera. Returns (3,) or (3,N) in camera frame."""
    return (R @ P.reshape(3, -1) + t).reshape(P.shape)


def build_cylindrical_panorama(
    images: Dict[str, np.ndarray],
    calibration: Dict[str, dict],
    order: Optional[List[str]] = None,
    cylinder_radius: float = 2.0,
    theta_range: tuple = (np.pi / 2, -np.pi / 2),
    z_range: tuple = (0.0, 2.5),
    out_width: int = 4000,
    out_height: int = 1200,
    blend_overlap: bool = True,
) -> np.ndarray:
    """
    Build a cylindrical panorama from calibrated camera images.

    Waymo vehicle frame: X forward, Y right, Z up. Cylinder axis = Z.
    Point on cylinder: P = (R*cos(theta), R*sin(theta), z). theta=0 is +X (front).
    Left-to-right panorama: theta from pi/2 (left) to -pi/2 (right). z is vertical.

    Args:
        images: camera_name -> BGR image
        calibration: camera_name -> {K, D, R, t, width, height}
        order: left-to-right camera order (default DEFAULT_PANORAMA_ORDER)
        cylinder_radius: radius in meters
        theta_range: (min_theta, max_theta) in radians (default pi/2 to -pi/2 = left to right)
        z_range: (min_z, max_z) in meters, vertical (Waymo Z up)
        out_width, out_height: output panorama size
        blend_overlap: if True, blend where multiple cameras see the same point

    Returns:
        Panorama BGR image (out_height x out_width x 3).
    """
    if order is None:
        order = list(DEFAULT_PANORAMA_ORDER)
    panorama = np.zeros((out_height, out_width, 3), dtype=np.float64)
    weight_sum = np.zeros((out_height, out_width), dtype=np.float64)

    theta_min, theta_max = theta_range
    z_min, z_max = z_range

    for cam_name in order:
        if cam_name not in images or cam_name not in calibration:
            continue
        img = images[cam_name]
        cal = calibration[cam_name]
        K = cal["K"]
        D = cal["D"]
        R = cal["R"]
        t = cal["t"]
        h_src, w_src = img.shape[:2]

        # Undistort the image so we can use pinhole projection
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w_src, h_src), 1.0)
        img_undist = cv2.undistort(img, K, D, None, K_new)
        K = K_new

        # For each output pixel (j, i) -> (theta, z). Waymo: X forward, Y right, Z up.
        for j in range(out_height):
            z = z_min + (z_max - z_min) * (j + 0.5) / out_height
            for i in range(out_width):
                theta = theta_min + (theta_max - theta_min) * (i + 0.5) / out_width
                P_vehicle = np.array([
                    cylinder_radius * np.cos(theta),
                    cylinder_radius * np.sin(theta),
                    z,
                ], dtype=np.float64)
                P_cam = _vehicle_to_camera_point(R, t, P_vehicle)
                if P_cam[2] <= 0:
                    continue
                u = K[0, 0] * P_cam[0] / P_cam[2] + K[0, 2]
                v = K[1, 1] * P_cam[1] / P_cam[2] + K[1, 2]
                if u < 0 or u >= w_src or v < 0 or v >= h_src:
                    continue
                # Bilinear sample
                u0, v0 = int(np.floor(u)), int(np.floor(v))
                u1, v1 = min(u0 + 1, w_src - 1), min(v0 + 1, h_src - 1)
                du, dv = u - u0, v - v0
                s00 = img_undist[v0, u0].astype(np.float64)
                s10 = img_undist[v0, u1].astype(np.float64)
                s01 = img_undist[v1, u0].astype(np.float64)
                s11 = img_undist[v1, u1].astype(np.float64)
                sample = (1 - du) * (1 - dv) * s00 + du * (1 - dv) * s10 + (1 - du) * dv * s01 + du * dv * s11
                if blend_overlap:
                    w = 1.0
                    panorama[j, i] += w * sample
                    weight_sum[j, i] += w
                else:
                    if weight_sum[j, i] == 0:
                        panorama[j, i] = sample
                        weight_sum[j, i] = 1.0
                    break

    # Normalize where we have coverage
    mask = weight_sum > 0
    panorama[mask] /= weight_sum[mask, np.newaxis]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    return panorama


def build_cylindrical_panorama_fast(
    images: Dict[str, np.ndarray],
    calibration: Dict[str, dict],
    order: Optional[List[str]] = None,
    cylinder_radius = 6.0,
    theta_range = (-2.1, 2.1),
    z_range = (-1.5, 3.0),
    out_width: int = 4000,
    out_height: int = 1200,
) -> tuple:
    """
    Vectorized version: build maps from (theta, y) to (u, v) per camera, then remap.
    Uses one camera per output pixel (first valid in order) to avoid blending cost.

    Returns:
        (panorama, cam_index_map) where cam_index_map is a uint8 array the same size
        as the panorama. Each pixel value is the index of the camera in `order` that
        contributed it (0 = first camera in order), or 255 if no camera covers it.
        Young's inpainting script can use this to mask any single camera's region.
    """
    if order is None:
        order = list(DEFAULT_PANORAMA_ORDER)
    theta_min, theta_max = theta_range
    z_min, z_max = z_range

    # Output coordinate grids. Waymo: X forward, Y right, Z up. Cylinder: P = (R*cos(theta), R*sin(theta), z)
    jj, ii = np.meshgrid(np.arange(out_height, dtype=np.float64), np.arange(out_width, dtype=np.float64), indexing="ij")
    theta = theta_min + (theta_max - theta_min) * (ii + 0.5) / out_width
    z = z_min + (z_max - z_min) * (jj + 0.5) / out_height
    P_x = cylinder_radius * np.cos(theta)
    P_y = cylinder_radius * np.sin(theta)
    P_vehicle = np.stack([P_x, P_y, z], axis=-1)

    panorama = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    filled = np.zeros((out_height, out_width), dtype=bool)
    cam_index_map = np.full((out_height, out_width), 255, dtype=np.uint8)

    for cam_idx, cam_name in enumerate(order):
        if cam_name not in images or cam_name not in calibration or filled.all():
            continue
        img = cv2.flip(images[cam_name], 1)  # mirror each image, keep same spots
        cal = calibration[cam_name]
        K_orig = cal["K"]
        D = cal["D"]
        R = cal["R"]   # vehicle-to-camera rotation
        t = cal["t"]   # vehicle-to-camera translation
        h_src, w_src = img.shape[:2]
        K_new, _ = cv2.getOptimalNewCameraMatrix(K_orig, D, (w_src, h_src), 1.0)
        img_undist = cv2.undistort(img, K_orig, D, None, K_new)

        # P_cam = R @ P_vehicle + t (row form: P_cam = (P_vehicle @ R.T) + t)
        P_cam = (P_vehicle @ R.T) + t.ravel()
        K = K_new
        # Waymo camera frame: X is forward (depth), Y right, Z down. Project: u = fx*Y/X + cx, v = fy*Z/X + cy
        depth = P_cam[:, :, 0]
        in_front = depth > 0.1
        depth_safe = np.maximum(depth, 1e-6)
        u = (K[0, 0] * P_cam[:, :, 1] / depth_safe + K[0, 2])
        v = (K[1, 1] * P_cam[:, :, 2] / depth_safe + K[1, 2])
        in_bounds = (u >= 0) & (u < w_src - 1) & (v >= 0) & (v < h_src - 1)
        use = in_front & in_bounds & ~filled
        if not np.any(use):
            continue
        map_u = u.astype(np.float32)
        map_v = v.astype(np.float32)
        remapped = cv2.remap(img_undist, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        panorama[use] = remapped[use]
        cam_index_map[use] = cam_idx
        filled[use] = True

    # Fix placement: first image ends up at last (right), last at first (left)
    panorama = cv2.flip(panorama, 1)
    cam_index_map = cv2.flip(cam_index_map, 1)
    return panorama, cam_index_map
