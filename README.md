# CMPE 297 – Final Project
## Gap Reconstruction in Autonomous Panoramic Vision

### Team – Group 1
- Young Suh – 015391571 (Inpainting / Reconstruction)
- Michael Khuri – 016531320 (Image Stitching / Integration)
- Lindsey Raven – 018200819 (Object Detection)

---

## Project Description

Autonomous vehicles use multiple strategically placed cameras to maintain a continuous understanding of their surroundings. However, gaps in information can still occur due to limited camera fields of view or camera failure. Such gaps in environmental perception can lead to critical failures and potential accidents, as the vehicle is unable to respond to unknown conditions.

The goal of this project is to develop a proof-of-concept pipeline that:
1. Stitches multi-camera Waymo images into a panorama
2. Simulates missing camera data by masking regions
3. Reconstructs missing regions using image inpainting
4. Evaluates whether object detection performance is preserved after reconstruction

---

## Repository Structure

```
├── image_stitching/      ← Michael: panorama generation from Waymo cameras
├── inpainting/           ← Young: gap reconstruction models
├── object_detection/     ← Lindsey: YOLOv8 detection baseline + evaluation
```

---

## Dataset Setup (All Members)

The dataset is **not stored in this repo** due to size. Each member must download it locally.

1. Go to https://waymo.com/open/download and sign in
2. Download **Perception Dataset v2.0.1 – modular without maps**
3. Download both:
   - `camera_image` parquet files
   - `camera_calibration` parquet files
4. Place them in your local copy of this folder:

```
image_stitching/
└── dataset/
    ├── camera_image/
    │   └── *.parquet
    └── camera_calibration/
        └── *.parquet
```

> The `camera_calibration` parquet for each segment must have the **same filename** as its corresponding `camera_image` parquet.

---

## Image Stitching (Michael)

Stitches 5 synchronized Waymo camera views (SIDE_LEFT, FRONT_LEFT, FRONT, FRONT_RIGHT, SIDE_RIGHT) into a single panorama using camera calibration data.

### Setup

```bash
cd image_stitching
pip install -r requirements.txt
```

### Run on all segments (recommended)

Process every parquet file in your dataset automatically:

```bash
# First frame of every segment (fast, good for testing)
python batch_stitch.py

# Every frame of every segment (full dataset)
python batch_stitch.py --all-frames
```

Outputs are saved to `outputs/<segment_name>/frame_0000.png`, etc.

---

### Run on a single frame

```bash
python run_stitch.py --parquet dataset/camera_image/<filename>.parquet --output outputs/panorama.png
```

Example:
```bash
python run_stitch.py --parquet dataset/camera_image/8993680275027614595_2520_000_2540_000.parquet --output outputs/panorama.png
```

### List available frames in a parquet file

```bash
python run_stitch.py --parquet dataset/camera_image/<filename>.parquet --list-frames
```

### Pick a specific frame by index

```bash
python run_stitch.py --parquet dataset/camera_image/<filename>.parquet --frame-index 5 --output outputs/panorama_frame5.png
```

### Save the 5 individual camera images alongside the panorama

```bash
python run_stitch.py --parquet dataset/camera_image/<filename>.parquet --output outputs/panorama.png --save-inputs inputs/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--parquet` | required | Path to camera_image parquet |
| `--output` | `panorama_output.png` | Output panorama path |
| `--frame-index` | `0` | Which frame to use (0 = first) |
| `--method` | `calibration` | `calibration` (recommended) or `homography` |
| `--list-frames` | — | Print available frame timestamps and exit |
| `--save-inputs` | — | Save 5 camera images to this folder |

---

## Data Sources

- **Waymo Open Dataset (Perception v2.0.1):** https://waymo.com/open/download
  - Multi-camera driving images
  - Camera calibration data (intrinsics + extrinsics)
  - Object detection annotations
