<p align="center">
  <img src="figures/readme_img.png" alt="Computer Vision Project Cover" width="100%">
</p>

# Computer-Vision-Project

## Authors

* Abrar Jawad Tarafder
* Ekambir Momi
* Thor Laski
* Antonio Lamacchia
* Emily Zelkowicz

# Wide-Baseline Feature Matching: SIFT vs. ORB

## Overview

This project compares two classical local feature matching algorithms, **Scale-Invariant Feature Transform (SIFT)** and **Oriented FAST and Rotated BRIEF (ORB)**, using the **HPatches** benchmark dataset. The objective is to evaluate each method's ability to establish reliable feature correspondences under changes in viewpoint and illumination while analyzing the trade-off between matching accuracy and computational efficiency.

Feature matching is a fundamental task in computer vision and serves as the foundation for applications such as image registration, panorama stitching, visual localization, Structure-from-Motion (SfM), and Simultaneous Localization and Mapping (SLAM). By evaluating SIFT and ORB under identical conditions, this project provides a direct comparison of two widely used handcrafted feature descriptors.

---

## Objectives

* Implement feature matching pipelines using **SIFT** and **ORB**.
* Evaluate both methods using the **HPatches** benchmark.
* Compare matching robustness under viewpoint and illumination changes.
* Measure matching accuracy and computational performance.
* Analyze the strengths and limitations of each approach.

---

## Methodology

The feature matching pipeline consists of the following stages:

1. Image preprocessing
2. Keypoint detection
3. Descriptor extraction
4. Descriptor matching
5. Match filtering using Lowe's ratio test
6. Geometric verification using ground-truth homographies
7. Performance evaluation

Both SIFT and ORB are evaluated using the same images, matching strategy, and evaluation criteria to ensure a fair comparison.

---

## Dataset

Experiments are conducted using the **HPatches** benchmark dataset, which contains image sequences exhibiting controlled **viewpoint** and **illumination** changes. Each sequence includes ground-truth homography matrices that enable objective evaluation of feature matching accuracy.

---

## Evaluation Metrics

The following metrics are used to compare the two methods:

* Precision
* Recall
* Mean Matching Accuracy (MMA)
* Reprojection Error
* Number of Detected Keypoints
* Number of Valid Matches
* Runtime

---

## Technologies

* Python
* OpenCV
* NumPy
* Matplotlib

---

## Repository Structure

```text
.
├── code/                  # Source code
├── data/                  # HPatches dataset
├── figures/               # Report figures and visualizations
├── report/                # CVPR-style report
├── README.md
```

---

