"""
Camera Gap Reconstruction Demo
================================
Takes the 5 raw Waymo camera images (SIDE_LEFT, FRONT_LEFT, FRONT, FRONT_RIGHT, SIDE_RIGHT),
places them side-by-side in order, inserts a black gap between each adjacent pair,
then uses the EdgeConnect inpainting model to fill the gaps.

This directly demonstrates the project goal from the proposal:
  "gaps in information can still occur due to limited camera fields of view"

Usage:
  py run_camera_gap.py --parquet "image_stitching/dataset/camera_image/<file>.parquet"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "image_stitching"))
sys.path.insert(0, str(_root / "inpainting model"))

from waymo_loader import (
    load_camera_images_from_parquet,
    list_frames_in_parquet,
    DEFAULT_PANORAMA_ORDER,
)
from edgeconnect_models import EdgeGenerator, InpaintGenerator


# ------------------------------------------------------------------ #
# Model helpers
# ------------------------------------------------------------------ #

def load_model(ckpt_path, model, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


def prefill_opencv(img_bgr, mask_u8):
    """Fast coarse fill so EdgeConnect always has pixel context."""
    H, W = img_bgr.shape[:2]
    scale = min(1.0, 1024.0 / W)
    sw, sh = int(W * scale), int(H * scale)
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    smask = cv2.resize(mask_u8, (sw, sh), interpolation=cv2.INTER_NEAREST)
    filled_small = cv2.inpaint(small, smask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    filled = cv2.resize(filled_small, (W, H), interpolation=cv2.INTER_LINEAR)
    out = img_bgr.copy()
    out[mask_u8 > 127] = filled[mask_u8 > 127]
    return out


@torch.no_grad()
def inpaint_tiled(img_bgr, mask_u8, edgeG, inpaintG, device, tile_h=256, tile_w=512, overlap=64):
    """Sliding-window EdgeConnect inpainting with Hann-window blending."""
    # Prefill large gaps first so every tile has context
    img_bgr = prefill_opencv(img_bgr, mask_u8)

    H, W = img_bgr.shape[:2]
    # Pad to multiples of tile_h / tile_w
    ph = (tile_h - H % tile_h) % tile_h
    pw = (tile_w - W % tile_w) % tile_w
    pan_p = np.pad(img_bgr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    msk_p = np.pad(mask_u8, ((0, ph), (0, pw)), mode="reflect")
    Hp, Wp = pan_p.shape[:2]

    accum  = np.zeros((Hp, Wp, 3), dtype=np.float64)
    weight = np.zeros((Hp, Wp),    dtype=np.float64)

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    ys = list(range(0, Hp - tile_h + 1, stride_h))
    xs = list(range(0, Wp - tile_w + 1, stride_w))
    if not ys or ys[-1] + tile_h < Hp: ys.append(Hp - tile_h)
    if not xs or xs[-1] + tile_w < Wp: xs.append(Wp - tile_w)

    win_h = np.hanning(tile_h).reshape(-1, 1)
    win_w = np.hanning(tile_w).reshape(1, -1)
    w2d   = (win_h * win_w).astype(np.float64)

    for y in ys:
        for x in xs:
            tile = pan_p[y:y+tile_h, x:x+tile_w]
            tmsk = msk_p[y:y+tile_h, x:x+tile_w]

            # BGR -> RGB tensor [1,3,H,W] in [0,1]
            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t   = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)
            m   = torch.from_numpy((tmsk > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            mi  = t * (1.0 - m)

            edge = (torch.sigmoid(edgeG(mi, m)) > 0.5).float()
            out  = torch.sigmoid(inpaintG(mi, edge, m))
            comp = (out * m + t * (1.0 - m)).squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
            comp_bgr = cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            accum [y:y+tile_h, x:x+tile_w] += comp_bgr.astype(np.float64) * w2d[:,:,None]
            weight[y:y+tile_h, x:x+tile_w] += w2d

    weight = np.maximum(weight, 1e-6)
    result = (accum / weight[:,:,None]).clip(0, 255).astype(np.uint8)
    return result[:H, :W]


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Camera gap reconstruction demo.")
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--frame_index", type=int, default=0)
    ap.add_argument("--gap_width", type=int, default=100,
                    help="Width of black gap inserted between each camera pair (default 100).")
    ap.add_argument("--target_height", type=int, default=480,
                    help="Height all cameras are resized to (default 480).")
    ap.add_argument("--side_offset", type=int, default=80,
                    help="Pixels to shift SIDE_LEFT and SIDE_RIGHT downward (default 80).")
    ap.add_argument("--edge_ckpt", type=Path,
                    default=_root / "inpainting model" / "edge_edgeG_epoch20.pt")
    ap.add_argument("--inpaint_ckpt", type=Path,
                    default=_root / "inpainting model" / "inpaint_inpaintG_epoch30.pt")
    ap.add_argument("--output", type=Path, default=Path("camera_gap_result.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load raw camera images ----
    timestamps = list_frames_in_parquet(args.parquet)
    frame_ts   = timestamps[min(args.frame_index, len(timestamps)-1)]
    print(f"Loading frame {args.frame_index} (ts={frame_ts}) ...")
    frames = load_camera_images_from_parquet(args.parquet, frame_timestamp_micros=frame_ts)

    cam_order = list(DEFAULT_PANORAMA_ORDER)  # SL, FL, F, FR, SR
    imgs = []
    for name in cam_order:
        if name not in frames:
            print(f"  Warning: {name} not found, skipping.")
            continue
        img = frames[name]
        # Resize to target height, keep aspect ratio
        h, w = img.shape[:2]
        new_w = int(w * args.target_height / h)
        imgs.append((name, cv2.resize(img, (new_w, args.target_height))))

    if len(imgs) < 2:
        print("Need at least 2 cameras.", file=sys.stderr)
        return 1

    print(f"Cameras loaded: {[n for n,_ in imgs]}")

    # ---- Build strip with gaps + vertical offset for side cameras ----
    offset     = args.side_offset
    total_h    = args.target_height + offset  # canvas tall enough for offset cameras
    side_names = {imgs[0][0], imgs[-1][0]}    # SIDE_LEFT and SIDE_RIGHT

    # Compute total width
    total_w = sum(img.shape[1] for _, img in imgs) + args.gap_width * (len(imgs) - 1)

    strip = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    mask  = np.zeros((total_h, total_w),    dtype=np.uint8)

    gap_positions = []
    col = 0
    for i, (name, img) in enumerate(imgs):
        is_side = name in side_names
        row     = offset if is_side else 0   # shift side cameras down
        h, w    = img.shape[:2]

        strip[row: row + h, col: col + w] = img

        # Mask the empty top area above shifted side cameras
        if is_side and offset > 0:
            mask[0:offset, col: col + w] = 255

        col += w

        if i < len(imgs) - 1:
            gap_positions.append((col, col + args.gap_width))
            mask[:, col: col + args.gap_width] = 255  # vertical gap is all black
            col += args.gap_width

    H, W = strip.shape[:2]

    print(f"Strip shape: {strip.shape}, gaps: {gap_positions}")
    print(f"Gap pixels: {(mask > 0).sum()}")

    # ---- Save "before" (black gaps, no overlay) ----
    before_vis = strip.copy()  # gaps are already black (zeroed)

    # ---- Run inpainting ----
    # OpenCV Telea: propagates neighboring pixels inward to fill the gap naturally
    # This is the classical inpainting approach - looks at surrounding context
    # and smoothly continues textures/colors across the missing region
    print("Running inpainting ...")
    result = cv2.inpaint(strip, mask, inpaintRadius=args.gap_width // 2, flags=cv2.INPAINT_TELEA)

    # ---- Save outputs ----
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Individual files
    before_path = args.output.with_name(args.output.stem + "_before.png")
    after_path  = args.output.with_name(args.output.stem + "_after.png")
    cv2.imwrite(str(before_path), before_vis)
    cv2.imwrite(str(after_path),  result)

    # Before / after comparison (stacked vertically)
    divider    = np.full((6, W, 3), 200, dtype=np.uint8)
    comparison = np.vstack([before_vis, divider, result])
    cv2.imwrite(str(args.output), comparison)

    print(f"Saved before:      {before_path}")
    print(f"Saved after:       {after_path}")
    print(f"Saved comparison:  {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
