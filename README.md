# CMPE 297 – Final Project
## Gap Reconstruction in Autonomous Panoramic Vision

### Team – Group 1
- Young Suh – 015391571
- Michael Khuri – 016531320
- Lindsey Raven – 018200819

---

# Project Description

Autonomous vehicles rely on multiple cameras to perceive their surroundings. However, perception gaps can occur due to limited camera field of view or camera failure. Missing visual information can reduce the ability of the vehicle to respond to unexpected conditions.

The goal of this project is to develop a proof-of-concept system that reconstructs missing visual regions in panoramic driving scenes. Multi-camera images will be stitched into panoramic views, and artificial gaps will be introduced to simulate missing visual information.

AI-based reconstruction models such as image inpainting or masked autoencoders will then be used to fill in the missing regions. The reconstructed images will be evaluated to determine how reconstruction quality affects downstream perception tasks such as object detection.

---

# Data Sources

### Waymo Open Dataset (Perception Dataset v2.0.1)

Dataset link:  
https://waymo.com/intl/jp/open/download

Data used in this project:

- Multi-camera driving images  
- Camera calibration data  
- Object detection annotations  

The dataset will be used to construct panoramic driving scenes and simulate missing camera information.

Due to the large dataset size, the data is **not stored in this repository**.
