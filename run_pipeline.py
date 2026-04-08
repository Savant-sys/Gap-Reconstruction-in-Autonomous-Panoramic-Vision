"""
End-to-end pipeline: stitch Waymo cameras into a panorama, detect gap regions,
and inpaint them with the trained EdgeConnect model.

Two modes:
  1. From parquet (stitch then inpaint):
       python run_pipeline.py --parquet dataset/camera_image/<file>.parquet
           --edge_ckpt "inpainting model/edge_edgeG_epoch20.pt"
           --inpaint_ckpt "inpainting model/inpaint_inpaintG_epoch30.pt"

  2. From existing panorama + cammap (skip stitching):
       python run_pipeline.py --panorama panorama.png --cammap panorama_cammap.png
           --edge_ckpt "inpainting model/edge_edgeG_epoch20.pt"
           --inpaint_ckpt "inpainting model/inpaint_inpaintG_epoch30.pt"

The gap mask is derived from the camera-index map (pixels with value 255 = no camera
coverage). To simulate a dropped camera, use --drop_cam <name> (e.g. FRONT).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# --- path setup so modules in subdirectories can be imported ---
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "image_stitching"))
sys.path.insert(0, str(_root / "inpainting model"))

from waymo_loader import (
    load_camera_images_from_parquet,
    load_camera_calibration,
    list_frames_in_parquet,
    DEFAULT_PANORAMA_ORDER,
)
from cylindrical_stitch import build_cylindrical_panorama_fast
from edgeconnect_models import EdgeGenerator, InpaintGenerator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: str, model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Load a checkpoint that may be a bare state_dict or wrapped in {"model": ...}."""
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


def bgr_to_rgb_tensor(bgr: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 BGR -> 1x3xHxW float32 [0,1] RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def rgb_tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """1x3xHxW float32 [0,1] -> HxWx3 uint8 BGR."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def mask_to_tensor(mask_uint8: np.ndarray) -> torch.Tensor:
    """HxW uint8 (0 or 255) -> 1x1xHxW float32 {0,1}."""
    m = (mask_uint8 > 127).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)


def pad_to_multiple(img: np.ndarray, multiple: int = 8):
    """Pad image (HxWx3 or HxW) to next multiple of `multiple`, return (padded, (h, w))."""
    h, w = img.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if img.ndim == 3:
        padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    else:
        padded = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    return padded, (h, w)


def prefill_with_opencv(panorama_bgr: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    """
    Use OpenCV Navier-Stokes inpainting to coarsely pre-fill large gaps.
    This gives the EdgeConnect model pixel context even in fully-masked tiles.
    Works on a downscaled copy for speed, then upscales back.
    """
    H, W = panorama_bgr.shape[:2]
    scale = min(1.0, 1024.0 / W)
    small_h, small_w = int(H * scale), int(W * scale)

    small_pan = cv2.resize(panorama_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_msk = cv2.resize(mask_uint8, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    # OpenCV inpaint: mask must be uint8 where >0 = inpaint
    prefilled_small = cv2.inpaint(small_pan, small_msk, inpaintRadius=5, flags=cv2.INPAINT_NS)
    prefilled = cv2.resize(prefilled_small, (W, H), interpolation=cv2.INTER_LINEAR)

    # Only replace gap pixels with the pre-fill; keep known pixels unchanged
    gap = mask_uint8 > 127
    result = panorama_bgr.copy()
    result[gap] = prefilled[gap]
    return result


@torch.no_grad()
def inpaint_tiled(
    panorama_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    edgeG: torch.nn.Module,
    inpaintG: torch.nn.Module,
    device: str,
    tile_h: int = 256,
    tile_w: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Inpaint a large panorama using a sliding-window tile approach.

    First pre-fills large gaps with OpenCV inpainting so every tile has
    some context, then refines with EdgeConnect.

    panorama_bgr : HxWx3 uint8 BGR — the stitched panorama
    mask_uint8   : HxW uint8, 255 = gap pixel to fill, 0 = known pixel
    Returns      : HxWx3 uint8 BGR inpainted panorama
    """
    print("  Pre-filling large gaps with OpenCV Navier-Stokes inpainting ...")
    panorama_bgr = prefill_with_opencv(panorama_bgr, mask_uint8)
    H, W = panorama_bgr.shape[:2]

    # Pad so tiles always fit evenly
    pan_pad, (orig_h, orig_w) = pad_to_multiple(panorama_bgr, tile_h)
    msk_pad, _ = pad_to_multiple(mask_uint8, tile_h)
    H_pad, W_pad = pan_pad.shape[:2]

    accum = np.zeros((H_pad, W_pad, 3), dtype=np.float64)
    weight = np.zeros((H_pad, W_pad), dtype=np.float64)

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    ys = list(range(0, H_pad - tile_h + 1, stride_h))
    xs = list(range(0, W_pad - tile_w + 1, stride_w))
    # Ensure we always cover the last strip
    if ys[-1] + tile_h < H_pad:
        ys.append(H_pad - tile_h)
    if xs[-1] + tile_w < W_pad:
        xs.append(W_pad - tile_w)

    total_tiles = len(ys) * len(xs)
    print(f"  Inpainting {total_tiles} tiles ({tile_h}x{tile_w}, overlap={overlap}) on {device} ...")

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tile_bgr = pan_pad[y: y + tile_h, x: x + tile_w]
            tile_msk = msk_pad[y: y + tile_h, x: x + tile_w]

            # Convert BGR -> RGB tensor
            tile_rgb_t = bgr_to_rgb_tensor(tile_bgr).to(device)    # [1,3,H,W]
            mask_t = mask_to_tensor(tile_msk).to(device)            # [1,1,H,W]
            masked_t = tile_rgb_t * (1.0 - mask_t)                  # zero out gaps

            # EdgeConnect forward pass
            edge_logits = edgeG(masked_t, mask_t)
            edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()

            out_logits = inpaintG(masked_t, edge_pred, mask_t)
            out = torch.sigmoid(out_logits)

            # Composite: keep known pixels from original, fill gaps with prediction
            comp = out * mask_t + tile_rgb_t * (1.0 - mask_t)

            comp_bgr = rgb_tensor_to_bgr(comp)  # HxWx3 uint8

            # Hann-window blend weight for smooth tile edges
            win_h = np.hanning(tile_h).reshape(-1, 1)
            win_w = np.hanning(tile_w).reshape(1, -1)
            w2d = (win_h * win_w).astype(np.float64)

            accum[y: y + tile_h, x: x + tile_w] += comp_bgr.astype(np.float64) * w2d[:, :, None]
            weight[y: y + tile_h, x: x + tile_w] += w2d

    # Normalise (avoid div by zero on unvisited padding)
    weight = np.maximum(weight, 1e-6)
    result = (accum / weight[:, :, None]).clip(0, 255).astype(np.uint8)
    return result[:orig_h, :orig_w]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end panorama gap inpainting pipeline."
    )

    # Input: either parquet OR pre-stitched panorama
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet", type=Path, help="Path to camera_image parquet file.")
    src.add_argument("--panorama", type=Path, help="Path to existing stitched panorama PNG.")

    ap.add_argument(
        "--cammap", type=Path, default=None,
        help="Camera-index map PNG (required when --panorama is used; auto-detected otherwise).",
    )
    ap.add_argument(
        "--calibration", type=Path, default=None,
        help="camera_calibration parquet (auto-detected from --parquet name if omitted).",
    )
    ap.add_argument("--frame_index", type=int, default=0, help="Frame index in parquet (default 0).")

    # Model checkpoints
    ap.add_argument(
        "--edge_ckpt", type=Path,
        default=_root / "inpainting model" / "edge_edgeG_epoch20.pt",
        help="EdgeGenerator checkpoint (.pt).",
    )
    ap.add_argument(
        "--inpaint_ckpt", type=Path,
        default=_root / "inpainting model" / "inpaint_inpaintG_epoch30.pt",
        help="InpaintGenerator checkpoint (.pt).",
    )

    # Gap mask control
    ap.add_argument(
        "--drop_cam", type=str, default=None,
        choices=list(DEFAULT_PANORAMA_ORDER) + ["ALL_GAPS"],
        help="Simulate dropping this camera.",
    )
    ap.add_argument(
        "--seam_gap", type=str, default=None,
        metavar="CAM1,CAM2",
        help="Create a vertical gap at the seam between two adjacent cameras "
             "(e.g. FRONT_LEFT,FRONT). Comma-separate multiple pairs with semicolons: "
             "'SIDE_LEFT,FRONT_LEFT;FRONT_RIGHT,SIDE_RIGHT'",
    )
    ap.add_argument(
        "--seam_gap_width", type=int, default=80,
        help="Width in pixels of each seam gap (default 80).",
    )

    # Tiling / output
    ap.add_argument("--tile_h", type=int, default=256)
    ap.add_argument("--tile_w", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument(
        "--output", type=Path, default=Path("pipeline_output.png"),
        help="Output path for the inpainted panorama.",
    )
    ap.add_argument("--save_intermediate", action="store_true",
                    help="Also save the raw stitched panorama and gap mask.")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Step 1: Get stitched panorama + camera-index map
    # ------------------------------------------------------------------ #
    if args.parquet is not None:
        parquet_path: Path = args.parquet
        if not parquet_path.is_file():
            print(f"Error: parquet not found: {parquet_path}", file=sys.stderr)
            return 1

        cal_path = args.calibration
        if cal_path is None:
            cal_path = _root / "image_stitching" / "dataset" / "camera_calibration" / parquet_path.name
        if not cal_path.is_file():
            # fallback: same filename in sibling dataset folder
            cal_path = parquet_path.parent.parent / "camera_calibration" / parquet_path.name
        if not cal_path.is_file():
            print(f"Error: calibration parquet not found. Try --calibration PATH", file=sys.stderr)
            return 1

        timestamps = list_frames_in_parquet(parquet_path)
        if not timestamps:
            print("Error: no frames in parquet.", file=sys.stderr)
            return 1
        idx = min(args.frame_index, len(timestamps) - 1)
        frame_ts = timestamps[idx]
        print(f"Loading frame {idx} (ts={frame_ts}) from {parquet_path.name} ...")

        images = load_camera_images_from_parquet(parquet_path, frame_timestamp_micros=frame_ts)
        calibration = load_camera_calibration(cal_path)

        print("Stitching panorama ...")
        panorama_bgr, cam_index_map = build_cylindrical_panorama_fast(
            images=images, calibration=calibration
        )

        if args.save_intermediate:
            stitch_path = args.output.with_name(args.output.stem + "_stitched.png")
            cammap_path = args.output.with_name(args.output.stem + "_cammap.png")
            cv2.imwrite(str(stitch_path), panorama_bgr)
            cv2.imwrite(str(cammap_path), cam_index_map)
            print(f"Saved stitched panorama: {stitch_path}")
            print(f"Saved camera map:        {cammap_path}")

    else:
        # Load pre-stitched panorama
        panorama_bgr = cv2.imread(str(args.panorama))
        if panorama_bgr is None:
            print(f"Error: could not read {args.panorama}", file=sys.stderr)
            return 1

        if args.cammap is None:
            # Try to auto-detect: same stem + _cammap.png
            auto = args.panorama.with_name(args.panorama.stem + "_cammap.png")
            if auto.is_file():
                args.cammap = auto
            else:
                print("Error: --cammap required when using --panorama (or name it <stem>_cammap.png).",
                      file=sys.stderr)
                return 1

        cam_index_map = cv2.imread(str(args.cammap), cv2.IMREAD_GRAYSCALE)
        if cam_index_map is None:
            print(f"Error: could not read cammap {args.cammap}", file=sys.stderr)
            return 1

    print(f"Panorama shape: {panorama_bgr.shape}")

    # ------------------------------------------------------------------ #
    # Step 1b: Crop to valid coverage band (remove uncoverable black sky/ground)
    # ------------------------------------------------------------------ #
    valid_pixels = (cam_index_map != 255)
    valid_rows = np.where(valid_pixels.any(axis=1))[0]
    valid_cols = np.where(valid_pixels.any(axis=0))[0]

    if len(valid_rows) == 0:
        print("Error: no valid camera coverage found in cammap.", file=sys.stderr)
        return 1

    r0, r1 = int(valid_rows[0]), int(valid_rows[-1]) + 1
    c0, c1 = int(valid_cols[0]), int(valid_cols[-1]) + 1
    panorama_bgr = panorama_bgr[r0:r1, c0:c1]
    cam_index_map = cam_index_map[r0:r1, c0:c1]
    print(f"Cropped to valid region: rows {r0}:{r1}, cols {c0}:{c1} -> shape {panorama_bgr.shape}")

    if args.save_intermediate:
        crop_path = args.output.with_name(args.output.stem + "_cropped.png")
        cv2.imwrite(str(crop_path), panorama_bgr)
        print(f"Saved cropped panorama: {crop_path}")

    # ------------------------------------------------------------------ #
    # Step 2: Build gap mask
    # ------------------------------------------------------------------ #
    cam_order = list(DEFAULT_PANORAMA_ORDER)

    if args.seam_gap is not None:
        # --- Seam gap mode: cut vertical strips at one or more camera seams ---
        gap_mask = np.zeros(panorama_bgr.shape[:2], dtype=np.uint8)

        # Support multiple pairs separated by semicolons: "SL,FL;FR,SR"
        pairs_raw = [s.strip() for s in args.seam_gap.split(";") if s.strip()]
        for pair_str in pairs_raw:
            parts = [p.strip() for p in pair_str.split(",")]
            if len(parts) != 2 or parts[0] not in cam_order or parts[1] not in cam_order:
                print(f"Error: each seam pair must be CAM1,CAM2 from {cam_order}", file=sys.stderr)
                return 1
            id1, id2 = cam_order.index(parts[0]), cam_order.index(parts[1])

            cols_cam1 = np.where((cam_index_map == id1).any(axis=0))[0]
            cols_cam2 = np.where((cam_index_map == id2).any(axis=0))[0]
            if len(cols_cam1) == 0 or len(cols_cam2) == 0:
                print(f"Warning: {parts[0]} or {parts[1]} has no coverage, skipping.")
                continue

            right_of_cam1 = int(cols_cam1.max())
            left_of_cam2  = int(cols_cam2.min())
            seam_col = (right_of_cam1 + left_of_cam2) // 2
            half = args.seam_gap_width // 2
            col_start = max(0, seam_col - half)
            col_end   = min(panorama_bgr.shape[1], seam_col + half)

            strip_mask = np.zeros(panorama_bgr.shape[:2], dtype=np.uint8)
            strip_mask[:, col_start:col_end] = 255
            valid_rows = (cam_index_map[:, col_start:col_end] != 255).any(axis=1)
            strip_mask[~valid_rows, col_start:col_end] = 0
            gap_mask = np.maximum(gap_mask, strip_mask)

            print(f"  Seam gap {parts[0]} | {parts[1]}: cols {col_start}:{col_end} "
                  f"(width={col_end - col_start}), gap pixels: {(strip_mask > 0).sum()}")

        print(f"Total gap pixels: {(gap_mask > 0).sum()}")

    elif args.drop_cam is not None and args.drop_cam != "ALL_GAPS":
        if args.drop_cam not in cam_order:
            print(f"Error: unknown camera '{args.drop_cam}'. Choose from {cam_order}.", file=sys.stderr)
            return 1
        cam_id = cam_order.index(args.drop_cam)
        gap_mask = (cam_index_map == cam_id).astype(np.uint8) * 255
        print(f"Simulating dropped camera: {args.drop_cam} (id={cam_id}), "
              f"gap pixels: {(gap_mask > 0).sum()}")
    else:
        # Real uncovered gaps within the valid band (seams between cameras)
        gap_mask = (cam_index_map == 255).astype(np.uint8) * 255
        print(f"Using real gap region (cammap==255), gap pixels: {(gap_mask > 0).sum()}")

    if (gap_mask > 0).sum() == 0:
        print("No gap pixels found. Saving panorama unchanged.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), panorama_bgr)
        return 0

    if args.save_intermediate:
        mask_path = args.output.with_name(args.output.stem + "_mask.png")
        cv2.imwrite(str(mask_path), gap_mask)
        print(f"Saved gap mask: {mask_path}")

    # ------------------------------------------------------------------ #
    # Step 3: Load EdgeConnect models
    # ------------------------------------------------------------------ #
    for attr, label in [("edge_ckpt", "EdgeGenerator"), ("inpaint_ckpt", "InpaintGenerator")]:
        p: Path = getattr(args, attr)
        if not p.is_file():
            print(f"Error: {label} checkpoint not found: {p}", file=sys.stderr)
            print("  Train with train_edgeconnect_gan.py or train_edgeconnect_baseline.py first.",
                  file=sys.stderr)
            return 1

    print("Loading models ...")
    edgeG = load_model(str(args.edge_ckpt), EdgeGenerator(), device)
    inpaintG = load_model(str(args.inpaint_ckpt), InpaintGenerator(), device)

    # ------------------------------------------------------------------ #
    # Step 4: Tiled inpainting
    # ------------------------------------------------------------------ #
    result = inpaint_tiled(
        panorama_bgr, gap_mask, edgeG, inpaintG, device,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        overlap=args.overlap,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Save result + before/after comparison
    # ------------------------------------------------------------------ #
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result)
    print(f"Saved inpainted panorama: {args.output} (shape {result.shape})")

    # Build a before/after comparison strip (cropped panorama on top, result on bottom)
    # Show the gap region on the "before" image in red so it's visible
    before_vis = panorama_bgr.copy()
    gap_vis = gap_mask > 127
    before_vis[gap_vis] = (before_vis[gap_vis].astype(np.int32) // 2 + np.array([0, 0, 128])).clip(0, 255).astype(np.uint8)

    divider = np.full((6, result.shape[1], 3), 200, dtype=np.uint8)  # grey line separator
    comparison = np.vstack([before_vis, divider, result])
    compare_path = args.output.with_name(args.output.stem + "_comparison.png")
    cv2.imwrite(str(compare_path), comparison)
    print(f"Saved before/after comparison: {compare_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
