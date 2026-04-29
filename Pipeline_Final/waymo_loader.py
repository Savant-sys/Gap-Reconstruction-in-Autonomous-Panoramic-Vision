"""
Waymo Open Dataset (Perception v2) loader for camera images.

Loads synchronized multi-camera frames from parquet files. Camera names in the
dataset are stored as integers; this module maps them to standard names and
returns images as OpenCV BGR arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import cv2


# Waymo CameraName enum:
# 0=UNKNOWN, 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
CAMERA_NAME_MAP = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

# Left-to-right order for panoramas
DEFAULT_PANORAMA_ORDER = [
    "SIDE_LEFT",
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
]

# Camera image parquet columns
COL_FRAME_TS = "key.frame_timestamp_micros"
COL_CAMERA_NAME = "key.camera_name"
COL_IMAGE = "[CameraImageComponent].image"

# Calibration parquet columns
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
    Load images for one frame from Waymo camera_image parquet.

    Returns:
        Dict[str, image_bgr]
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    table = pq.read_table(
        parquet_path,
        columns=[COL_FRAME_TS, COL_CAMERA_NAME, COL_IMAGE],
    )

    ts_col = table.column(COL_FRAME_TS)
    cam_col = table.column(COL_CAMERA_NAME)
    img_col = table.column(COL_IMAGE)

    if frame_timestamp_micros is None:
        target_ts = int(ts_col[0].as_py())
    else:
        target_ts = int(frame_timestamp_micros)

    result = {}

    for i in range(table.num_rows):
        ts = int(ts_col[i].as_py())

        if ts != target_ts:
            continue

        cam_code = int(cam_col[i].as_py())
        cam_name = CAMERA_NAME_MAP.get(cam_code)

        if cam_name is None:
            continue

        if camera_names is not None and cam_name not in camera_names:
            continue

        cell = img_col[i]

        if hasattr(cell, "as_py"):
            raw = cell.as_py()
        else:
            raw = bytes(cell)
        arr = np.frombuffer(raw, dtype=np.uint8)

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is not None:
            result[cam_name] = img

    return result


def list_frames_in_parquet(parquet_path: str | Path) -> List[int]:
    """
    Return sorted frame timestamps available in parquet file.
    """
    table = pq.read_table(parquet_path, columns=[COL_FRAME_TS])

    ts_col = table.column(COL_FRAME_TS)

    timestamps = sorted(
        set(int(ts_col[i].as_py()) for i in range(table.num_rows))
    )

    return timestamps


def get_panorama_order() -> List[str]:
    return list(DEFAULT_PANORAMA_ORDER)


def load_camera_calibration(
    calibration_parquet_path: str | Path,
) -> Dict[str, dict]:
    """
    Load camera calibration.

    Returns:
        {
          cam_name: {
             K, D, R, t, width, height
          }
        }

    K = intrinsic matrix
    D = distortion coeffs
    R = vehicle-to-camera rotation
    t = vehicle-to-camera translation
    """
    path = Path(calibration_parquet_path)

    if not path.exists():
        raise FileNotFoundError(path)

    table = pq.read_table(path)

    cam_col = table.column(COL_CALIB_CAMERA)

    result = {}

    for i in range(table.num_rows):
        code = int(cam_col[i].as_py())
        name = CAMERA_NAME_MAP.get(code)

        if name is None:
            continue

        fu = float(table.column(COL_CALIB_FU)[i].as_py())
        fv = float(table.column(COL_CALIB_FV)[i].as_py())
        cu = float(table.column(COL_CALIB_CU)[i].as_py())
        cv = float(table.column(COL_CALIB_CV)[i].as_py())

        K = np.array(
            [
                [fu, 0, cu],
                [0, fv, cv],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        D = np.array(
            [
                float(table.column(COL_CALIB_K1)[i].as_py()),
                float(table.column(COL_CALIB_K2)[i].as_py()),
                float(table.column(COL_CALIB_P1)[i].as_py()),
                float(table.column(COL_CALIB_P2)[i].as_py()),
                float(table.column(COL_CALIB_K3)[i].as_py()),
            ],
            dtype=np.float64,
        )

        ext = table.column(COL_CALIB_EXTR)[i]

        T = np.array(
            [float(ext[j].as_py()) for j in range(16)],
            dtype=np.float64,
        ).reshape(4, 4)

        # Waymo stores camera-to-vehicle transform
        # Need vehicle-to-camera
        R_c2v = T[:3, :3]
        t_c2v = T[:3, 3:4]

        R = R_c2v.T
        t = -R @ t_c2v

        width = int(table.column(COL_CALIB_W)[i].as_py())
        height = int(table.column(COL_CALIB_H)[i].as_py())

        result[name] = {
            "K": K,
            "D": D,
            "R": R,
            "t": t,
            "width": width,
            "height": height,
        }

    return result