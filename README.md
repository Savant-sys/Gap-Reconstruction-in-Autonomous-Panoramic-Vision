# Gap-Reconstruction-in-Autonomous-Panoramic-Vision

## Data Sources

This project uses the Waymo Open Dataset to simulate missing visual context in autonomous driving scenes.

### Waymo Open Dataset (Perception Dataset v2.0.1)
https://waymo.com/intl/jp/open/download

The dataset contains:
- Multi-camera images from autonomous vehicles
- Camera calibration parameters
- Object detection annotations

For this project we will use:
- Front and side camera images
- Camera calibration data
- Object detection annotations

These images will be stitched into panoramic views. Artificial masking will then simulate missing camera input to test reconstruction models.

Due to dataset size, the dataset is not stored in this repository. Instructions for downloading it are available on the official Waymo website.
