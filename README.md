# CMPE 297 – Final Project
## Paper title: Stitch, Mask, Inpaint, Detect: An Evaluation Pipeline for Assessing Object Detection With Missing Visual Context in Multi-Camera Environment

### Team – Group 1
- Young Suh – 015391571 (Inpainting / Reconstruction)
- Michael Khuri – 016531320 (Image Stitching / Integration)
- Lindsey Raven – 018200819 (Object Detection)

---
## Image processing and Pipeline for our project

<img width="2271" height="2375" alt="image" src="https://github.com/user-attachments/assets/414f1cf0-af8e-4475-af5d-f1a7f711c570" />

## Project Objective

Gap reconstruction using image inpainting for generation of Autonomous Panoramic Vision

## Project Description

Autonomous vehicles use multiple strategically placed cameras to maintain a continuous understanding of their surroundings. However, gaps in information can still occur due to limited camera fields of view or camera failure. Such gaps in environmental perception can lead to critical failures and potential accidents, as the vehicle is unable to respond to unknown conditions.

The goal of this project is to develop a proof-of-concept pipeline that:
1. Stitches multi-camera Waymo images into a panorama
2. Simulates missing camera data by masking regions
3. Reconstructs missing regions using image inpainting
4. Evaluates whether object detection performance is preserved after reconstruction

---

## The pipeline (3 steps)

```
Raw camera images (parquet)
    → Step 1: Stitching                   → one panorama image per frame (Michael's part)
    → Step 2: Inpainting using LaMa       → panorama with one region masked, then filled in by a model (Young's part)
    → Step 3: Object detection using YOLO → run a detector on original vs reconstructed (Lindsey’s part)
```



---

## What you need before starting

- **Dataset aggregation:** Waymo Perception v2.0.1 (modular, no maps). Training data should be downloaded from training folder (https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_1/training?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))).
- Download the `camera_image` and `camera_calibration` parquet files from validation folder (https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_1/validation?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))) and put them in:
  - `image_stitching/dataset/camera_image/*.parquet`
  - `image_stitching/dataset/camera_calibration/*.parquet`  
  The calibration file for each segment must have the same filename as the camera_image file.

---

## Step 1: Stitching images to generate panoramas

**What it does:** Reads 5 camera images [side left|front left|front|front right|side right] from a parquet file and combines them into one wide image (panorama) using the calibration. It also saves a “cammap” image: same size as the panorama, but each pixel is a number 0–4 saying which camera that pixel came from (or 255 if no camera covered it). The inpainting step needs the cammap to know which region to mask.

**Where the code is:** `image_stitching/`

**Commands (run from repo root):**

```bash
cd image_stitching
pip install -r requirements.txt
```

Then either:

- **All segments, first frame only (good first run):**
  ```bash
  py batch_stitch.py
  ```
- **All segments, every frame:**
  ```bash
  py batch_stitch.py --all-frames
  ```

**Output:**  
`image_stitching/outputs/<segment_name>/frame_0000.png` (panorama) and `frame_0000_cammap.png` (which pixel came from which camera).  
One segment name looks like `10504764403039842352_460_000_480_000`.

So after Step 1 you have: a bunch of folders under `image_stitching/outputs/`, each with `frame_*.png` and `frame_*_cammap.png`.

---

## Step 2: Inpainting for filling the missing region in camera images using LaMa model

**What it does:** For each panorama we pretend one camera failed. We use the cammap to black out that camera’s region, then an AI model (EdgeConnect) fills it in. The result is a “reconstructed” panorama. We compare it to the original and compute PSNR/SSIM.

**Where the code is:** `inpainting model/` (folder with a space in the name). The model weights are the `.pt` files in that folder (e.g. `edge_edgeG_epoch20.pt`, `inpaint_inpaintG_epoch30.pt`).

**2a. Build the inpainting dataset**

The inpainting script does not read the stitching outputs directly. A separate script converts them into “image + mask + masked image” triplets and writes them to a folder. You run that once (or whenever you have new stitching outputs).

From **repo root**:

```bash
py scripts/build_inpainting_dataset.py --output-dir inpainting/waymo_data/masks
```

This reads `image_stitching/outputs/<segment>/frame_*.png` and `frame_*_cammap.png`, and for each frame and each camera index (0–4) creates one triplet: full image, binary mask for that camera, and image with that region blacked out. It writes:

- `inpainting/waymo_data/masks/images/`
- `inpainting/waymo_data/masks/masks/`
- `inpainting/waymo_data/masks/masked/`

So you must run **Step 1 first** so that `image_stitching/outputs/` has panoramas and cammaps. Then run this build script.

**2b. Run inpainting evaluation**

From **repo root** (or from `inpainting model`):

```bash
cd "inpainting model"
py eval_edgeconnect.py --root ../inpainting/waymo_data/masks --edge_ckpt edge_edgeG_epoch20.pt --inpaint_ckpt inpaint_inpaintG_epoch30.pt --save_images --save_dir ./eval_outputs
```

- `--root` points to the folder that contains `images/`, `masks/`, `masked/` (the one you built in 2a).
- `--edge_ckpt` and `--inpaint_ckpt` are the two generator checkpoint files (`.pt`) in `inpainting model/`.
- `--save_images` writes result images; `--save_dir` is where they go.

## Step 3: Classification and regression using YOLO11 model performed on inpainted images obtained at Step 2
Our group made an end-to-end model which fuses LaMa and YOLO11 for generating smoothly connected panoramic visions with large field of view.

From **repo root**

```bash
$PROJECT = "C:\Path\To\Gap-Reconstruction-in-Autonomous-Panoramic-Vision"

cd "$PROJECT\Pipeline_Final"

py run_full_waymo_gap_pipeline_dir.py `
  --camera_image_dir "$PROJECT\image_stitching\dataset\camera_image" `
  --camera_calibration_dir "$PROJECT\image_stitching\dataset\camera_calibration" `
  --output_root "full_pipeline_output_mask_0.5" `
  --max_segments 1 `
  --frames_per_segment 1 `
  --lama_root "$PROJECT\lama" `
  --lama_model_path "$PROJECT\lama\big-lama\big-lama" `
  --lama_checkpoint best.ckpt `
  --lama_gpu 0 `
  --yolo_model "yolo11x.pt" `
  --yolo_device 0 `
  --mask_percent 0.5 `
  --skip_existing

(GPU command)
--yolo_device 0
(CPU command)
--yolo_device cpu
```

Once it's working, you need to change from --max_segments 1 to --max_segments 0.

**Output:**  
`inpainting model/eval_outputs/` — comparison and visualization images (e.g. `frame_0000_cam0_comp.png`, `frame_0000_cam0_viz.png`). That’s where you “see the pics” after inpainting.

---

## Order of operations

1. Put Waymo parquet files in `image_stitching/dataset/` as above.
2. Run stitching: `cd image_stitching` then `py batch_stitch.py` (or `--all-frames`).
3. Build inpainting data: from repo root, `py scripts/build_inpainting_dataset.py --output-dir inpainting/waymo_data/masks`.
4. Run inpainting: `cd "inpainting model"` then the `eval_edgeconnect.py` command above with your `.pt` files and `--root ../inpainting/waymo_data/masks`.

---

## Where things live

| What | Where |
|------|--------|
| Raw data | `image_stitching/dataset/camera_image/` and `camera_calibration/` |
| Stitching output (panoramas + cammaps) | `image_stitching/outputs/<segment_name>/` |
| Inpainting dataset (images, masks, masked) | `inpainting/waymo_data/masks/` (created by the build script) |
| Inpainting result images | `inpainting model/eval_outputs/` (after you run eval with `--save_images`) |

---

## Single-frame stitching (optional)

To stitch one parquet file and one frame manually:

```bash
cd image_stitching
py run_stitch.py --parquet dataset/camera_image/<filename>.parquet --output outputs/panorama.png
```

That writes `outputs/panorama.png` and `outputs/panorama_cammap.png` for that one frame. For the full pipeline you normally use `batch_stitch.py` so all segments and frames go under `outputs/<segment_name>/`.
