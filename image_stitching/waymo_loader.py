"""
Waymo Open Dataset (Perception v2) loader for camera images.

Loads synchronized multi-camera frames from parquet files. Camera names in the
dataset are stored as integers; this module maps them to standard names and
returns images as OpenCV BGR arrays.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import cv2


# Waymo CameraName enum (from dataset.proto): 0=UNKNOWN, 1=FRONT, 2=FRONT_LEFT,
# 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
CAMERA_NAME_MAP = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

# Default order for left-to-right panorama (driver view: left side of car = left of panorama)
DEFAULT_PANORAMA_ORDER = [
    "SIDE_LEFT",
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
]

# Parquet column names used in Waymo Perception v2 camera_image tables
COL_FRAME_TS = "key.frame_timestamp_micros"
COL_CAMERA_NAME = "key.camera_name"
COL_IMAGE = "[CameraImageComponent].image"

# Camera calibration parquet column names (Perception v2)
COL_CALIB_CAMERA = "key.camera_name"
COL_CALIB_FU = "[CameraCalibrationComponent].intrinsic.f_u"
COL_CALIB_FV = "[CameraCalibrationComponent].intrinsic.f_v"
COL_CALIB_CU = "[CameraCalibrationComponent].intrinsic.c_u"
COL_CALIB_CV = "[CameraCalibrationComponent].intrinsic.c_v"
COL_CALIB_K1 = "[CameraCalibrationComponent].intrinsic.k1"
COL_CALIB_K2 = "[CameraCalibrationComponent].intrinsic.k2"
COL_CALIB_P1 = "[CameraCalibrationComponent].intrinsic.p1"
COL_CALIB_P2 = "[CameraCalibrationComponent].intrinsic.p2"
COL_CALIB_K3 = "[CameraCalibrationComponent].intrinsic.k3"
COL_CALIB_EXTR = "[CameraCalibrationComponent].extrinsic.transform"
COL_CALIB_W = "[CameraCalibrationComponent].width"
COL_CALIB_H = "[CameraCalibrationComponent].height"


def load_camera_images_from_parquet(
    parquet_path: str | Path,
    frame_timestamp_micros: Optional[int] = None,
    camera_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load images for one frame from a Waymo camera_image parquet file.

    Args:
        parquet_path: Path to the .parquet file (e.g. under dataset/camera_image/).
        frame_timestamp_micros: If given, load only this frame; otherwise the first
            timestamp in the file is used.
        camera_names: If given, load only these cameras (e.g. ["FRONT", "FRONT_LEFT"]).
            Must use string names as in CAMERA_NAME_MAP. If None, all cameras are loaded.

    Returns:
        Dictionary mapping camera name (str) to BGR image (numpy array, HxWx3).
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(
        parquet_path,
        columns=[COL_FRAME_TS, COL_CAMERA_NAME, COL_IMAGE],
    )

    ts_col = table.column(COL_FRAME_TS)
    cam_col = table.column(COL_CAMERA_NAME)
    img_col = table.column(COL_IMAGE)

    # Resolve frame timestamp
    if frame_timestamp_micros is not None:
        target_ts = frame_timestamp_micros
    else:
        target_ts = int(ts_col[0].as_py())

    # Reverse mapping: string name -> integer code
    name_to_code = {v: k for k, v in CAMERA_NAME_MAP.items()}
    if camera_names is not None:
        allowed_codes = {name_to_code[n] for n in camera_names}

    result: Dict[str, np.ndarray] = {}
    for i in range(table.num_rows):
        if int(ts_col[i].as_py()) != target_ts:
            continue
        code = int(cam_col[i].as_py())
        name = CAMERA_NAME_MAP.get(code)
        if name is None:
            continue
        if camera_names is not None and name not in camera_names:
            continue
        raw = bytes(img_col[i])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            result[name] = bgr

    return result


def list_frames_in_parquet(parquet_path: str | Path) -> List[int]:
    """
    Return sorted list of frame timestamps (micros) present in the parquet file.
    Useful for choosing a frame to stitch.
    """
    table = pq.read_table(parquet_path, columns=[COL_FRAME_TS])
    ts_col = table.column(COL_FRAME_TS)
    timestamps = sorted(set(int(ts_col[i].as_py()) for i in range(table.num_rows)))
    return timestamps


def get_panorama_order() -> List[str]:
    """Return the default camera order for left-to-right panorama."""
    return list(DEFAULT_PANORAMA_ORDER)


def load_camera_calibration(
    calibration_parquet_path: str | Path,
) -> Dict[str, dict]:
    """
    Load camera intrinsics and extrinsics from a Waymo camera_calibration parquet file.

    The parquet must match the segment of your camera_image file (same filename in
    dataset/camera_calibration/ as in dataset/camera_image/).

    Returns:
        Dict mapping camera name (str) to:
          - "K": 3x3 numpy float64 intrinsic matrix
          - "D": distortion coeffs (k1, k2, p1, p2, k3)
          - "R": 3x3 rotation vehicle-to-camera
          - "t": 3x1 translation vehicle-to-camera (meters)
          - "width", "height": image size
    """
    path = Path(calibration_parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    table = pq.read_table(path)
    result: Dict[str, dict] = {}
    cam_col = table.column(COL_CALIB_CAMERA)
    for i in range(table.num_rows):
        code = int(cam_col[i].as_py())
        name = CAMERA_NAME_MAP.get(code)
        if name is None:
            continue
        fu = float(table.column(COL_CALIB_FU)[i].as_py())
        fv = float(table.column(COL_CALIB_FV)[i].as_py())
        cu = float(table.column(COL_CALIB_CU)[i].as_py())
        cv = float(table.column(COL_CALIB_CV)[i].as_py())
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float64)
        D = np.array([
            float(table.column(COL_CALIB_K1)[i].as_py()),
            float(table.column(COL_CALIB_K2)[i].as_py()),
            float(table.column(COL_CALIB_P1)[i].as_py()),
            float(table.column(COL_CALIB_P2)[i].as_py()),
            float(table.column(COL_CALIB_K3)[i].as_py()),
        ], dtype=np.float64)
        ext = table.column(COL_CALIB_EXTR)[i]
        T = np.array([float(ext[j].as_py()) for j in range(16)], dtype=np.float64).reshape(4, 4)
        # Waymo stores camera-to-vehicle: P_vehicle = R_c2v @ P_cam + t_c2v. We need vehicle-to-camera for projection: P_cam = R_v2c @ P_vehicle + t_v2c.
        R_c2v = T[:3, :3].copy()
        t_c2v = T[:3, 3:4].copy()
        R = R_c2v.T
        t = -R @ t_c2v
        w = int(table.column(COL_CALIB_W)[i].as_py())
        h = int(table.column(COL_CALIB_H)[i].as_py())
        result[name] = {"K": K, "D": D, "R": R, "t": t, "width": w, "height": h}
    return result
