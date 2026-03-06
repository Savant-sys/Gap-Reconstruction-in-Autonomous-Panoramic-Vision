# CMPE 297 – Final Project
## Gap Reconstruction in Autonomous Panoramic Vision

### Team – Group 1
- Young Suh – 015391571
- Michael Khuri – 016531320
- Lindsey Raven – 018200819

---

# Project Description

Autonomous vehicles use multiple strategically placed cameras to maintain a continuous understanding of their surroundings. However, gaps in information can still occur due to limited camera fields of view or camera failure. Such gaps in environmental perception can lead to critical failures and potential accidents, as the vehicle is unable to respond to unknown conditions.

The goal of this project is to address this challenge by developing a proof-of-concept model that reconstructs a panoramic view from multiple camera images and fills missing regions or gaps. To simulate missing data, portions of the panoramic image will be artificially removed, after which AI models with purposes similar to masked autoencoding and image inpainting will be used to reconstruct the missing areas. As a part of this project, we will investigate what types of image context are necessary for accurate reconstruction and identify the threshold of information loss beyond which the model can no longer reliably recover the scene and the reconstructed output becomes unreliable.

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
