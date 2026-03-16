"""
Image stitching pipeline: feature detection, matching, homography estimation,
and warping to build a panorama from multiple overlapping camera views.

Designed to work with ordered camera images (e.g. from Waymo: SIDE_LEFT ->
FRONT_LEFT -> FRONT -> FRONT_RIGHT -> SIDE_RIGHT). Each new image is stitched
to the current panorama using keypoint matching and homography.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Feature detector configuration
# -----------------------------------------------------------------------------

def _create_detector(descriptor: str = "ORB", **kwargs) -> Tuple[cv2.Feature2D, int, bool]:
    """
    Create feature detector and return (detector, norm_type, cross_check).
    ORB uses NORM_HAMMING; SIFT/SURF use NORM_L2. BFMatcher crossCheck is
    typically True for 1-1 matches.
    """
    descriptor = descriptor.upper()
    if descriptor == "ORB":
        det = cv2.ORB_create(
            nfeatures=kwargs.get("nfeatures", 5000),
            scaleFactor=kwargs.get("scaleFactor", 1.2),
            nlevels=kwargs.get("nlevels", 8),
        )
        return det, cv2.NORM_HAMMING, True
    if descriptor == "SIFT":
        det = cv2.SIFT_create(
            nfeatures=kwargs.get("nfeatures", 5000),
            contrastThreshold=kwargs.get("contrastThreshold", 0.04),
            edgeThreshold=kwargs.get("edgeThreshold", 10),
        )
        return det, cv2.NORM_L2, True
    raise ValueError(f"Unknown descriptor: {descriptor}. Use 'ORB' or 'SIFT'.")


def _detect_and_compute(
    detector: cv2.Feature2D,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Detect keypoints and compute descriptors for one image."""
    return detector.detectAndCompute(image, mask)


def _match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    norm_type: int,
    cross_check: bool,
    ratio_thresh: float = 0.75,
    min_match_count: int = 10,
) -> List[cv2.DMatch]:
    """
    Match descriptors between two images. Uses ratio test (Lowe) when
    norm_type is L2; otherwise uses BFMatcher with crossCheck.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    if norm_type == cv2.NORM_L2 and not cross_check:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        return good
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def _estimate_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_reproj_thresh: float = 5.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Estimate homography from image1 to image2 using matched keypoints.
    src_pts are in image1, dst_pts in image2. Returns (H, inlier_mask).
    """
    if len(matches) < 4:
        return None, np.array([])
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh
    )
    if H is None:
        return None, np.array([])
    return H, mask.ravel()


def _warp_and_stitch(
    img_src: np.ndarray,
    img_dst: np.ndarray,
    H: np.ndarray,
    blend: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp img_src by H and composite onto img_dst. Assumes H maps src -> dst
    coords. Returns (stitched_bgr, mask of valid pixels: 0 or 255).
    """
    h1, w1 = img_src.shape[:2]
    h2, w2 = img_dst.shape[:2]

    # Corners of src in src coords; warp to dst coords
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    x_min = min(warped_corners[..., 0].min(), 0)
    x_max = max(warped_corners[..., 0].max(), w2)
    y_min = min(warped_corners[..., 1].min(), 0)
    y_max = max(warped_corners[..., 1].max(), h2)

    # Translation so canvas contains both: dst at (tx, ty), warped src in same frame
    tx, ty = -x_min, -y_min
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))

    # Combined transform: src -> dst coords -> canvas coords (H is 3x3 from findHomography)
    H_src_to_dst = np.asarray(H, dtype=np.float64).reshape(3, 3)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_src_to_canvas = T @ H_src_to_dst

    warped_src = cv2.warpPerspective(
        img_src, H_src_to_canvas, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Place dst on canvas at (tx, ty)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[int(ty) : int(ty) + h2, int(tx) : int(tx) + w2] = img_dst

    warped_mask = (warped_src.sum(axis=2) > 0).astype(np.uint8) * 255
    dst_region = np.zeros((out_h, out_w), dtype=np.uint8)
    dst_region[int(ty) : int(ty) + h2, int(tx) : int(tx) + w2] = 255

    if blend:
        overlap = (warped_mask > 0) & (dst_region > 0)
        canvas = canvas.astype(np.float32)
        warped_src_f = warped_src.astype(np.float32)
        canvas[overlap] = 0.5 * (canvas[overlap] + warped_src_f[overlap])
        canvas[~overlap & (warped_mask > 0)] = warped_src_f[~overlap & (warped_mask > 0)]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    else:
        canvas[warped_mask > 0] = warped_src[warped_mask > 0]

    return canvas, (canvas.sum(axis=2) > 0).astype(np.uint8) * 255


def stitch_two_images(
    img_left: np.ndarray,
    img_right: np.ndarray,
    descriptor: str = "ORB",
    min_match_count: int = 10,
    ransac_reproj_thresh: float = 5.0,
    blend: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Stitch two images: img_left is warped and composited with img_right so that
    the overlap aligns. Left is "source" (we warp it), right is "destination".

    Returns:
        (stitched_bgr, mask_uint8, num_inliers). If homography fails, (None, None, 0).
    """
    detector, norm_type, cross_check = _create_detector(descriptor)
    kp1, des1 = _detect_and_compute(detector, img_left)
    kp2, des2 = _detect_and_compute(detector, img_right)
    matches = _match_features(des1, des2, norm_type, cross_check, min_match_count=min_match_count)
    if len(matches) < min_match_count:
        return None, None, 0
    H, inlier_mask = _estimate_homography(kp1, kp2, matches, ransac_reproj_thresh)
    if H is None or inlier_mask.sum() < min_match_count:
        return None, None, int(inlier_mask.sum()) if inlier_mask.size else 0
    stitched, mask = _warp_and_stitch(img_left, img_right, H, blend=blend)
    return stitched, mask, int(inlier_mask.sum())


class PanoramaStitcher:
    """
    Stitch multiple images in a given order into one panorama. Each new image
    is matched and warped against the current panorama (or the previous image
    when building the first pair).
    """

    def __init__(
        self,
        descriptor: str = "ORB",
        min_match_count: int = 10,
        ransac_reproj_thresh: float = 5.0,
        blend: bool = False,
    ):
        self.descriptor = descriptor
        self.min_match_count = min_match_count
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.blend = blend

    def stitch(
        self,
        images: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Stitch a list of images in order (left to right). Each image is
        stitched to the accumulated panorama. Returns final panorama or None
        if stitching fails.
        """
        if not images:
            return None
        if len(images) == 1:
            return images[0].copy()

        panorama = images[0]
        for i in range(1, len(images)):
            panorama, mask, inliers = stitch_two_images(
                panorama,
                images[i],
                descriptor=self.descriptor,
                min_match_count=self.min_match_count,
                ransac_reproj_thresh=self.ransac_reproj_thresh,
                blend=self.blend,
            )
            if panorama is None:
                return None
        return panorama
