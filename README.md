<p align="center">
  <img src="figures/readme_img.png" alt="Computer Vision Project Cover" width="100%">
</p>

# EECS 4422: Computer Vision | Summer 2026 Project Report

# A Comparative Study of SIFT and ORB for Wide-Baseline Feature Matching

## Authors

- Abrar Jawad Tarafder
- Ekambir Momi
- Thor Laski
- Antonio Lamacchia
- Emily Zelkowicz

---

## Overview

This project presents a comparative evaluation of two classical local feature matching algorithms, **Scale-Invariant Feature Transform (SIFT)** and **Oriented FAST and Rotated BRIEF (ORB)**, using the **HPatches** benchmark dataset. The objective is to evaluate each method's ability to establish reliable feature correspondences under changes in viewpoint and illumination while examining the trade-off between matching robustness and computational efficiency.

Local feature matching is a fundamental task in computer vision and forms the basis of numerous applications, including image registration, panorama stitching, visual localization, Structure-from-Motion (SfM), simultaneous localization and mapping (SLAM), and 3D reconstruction. By evaluating SIFT and ORB under identical experimental conditions, this project provides a fair comparison of two widely used handcrafted feature extraction methods.

---

## Objectives

- Implement feature matching pipelines using **SIFT** and **ORB**.
- Evaluate both methods using the **HPatches** benchmark dataset.
- Compare feature matching robustness under viewpoint and illumination changes.
- Measure matching accuracy and computational performance.
- Analyze the trade-off between matching robustness and computational efficiency.

---

## Methodology

The feature matching pipeline consists of the following stages:

1. Image preprocessing
2. Feature extraction (SIFT / ORB)
3. Feature matching
4. Best correspondence selection using Lowe's ratio test
5. Affine transformation estimation
6. Geometric verification
7. Performance evaluation

Both SIFT and ORB are evaluated using the same image pairs, preprocessing pipeline, matching strategy, and evaluation criteria to ensure a fair comparison.

---

## Dataset

Experiments are conducted using the **HPatches** benchmark dataset, which contains image sequences exhibiting controlled **viewpoint** and **illumination** changes. Each sequence includes ground-truth homography matrices that enable objective evaluation of feature matching performance.

---

## Evaluation Metrics

The following metrics are used to compare the two methods:

- Number of Detected Keypoints
- Precision
- Recall
- Mean Matching Accuracy (MMA)
- Reprojection Error
- Runtime

---

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Repository Structure

```text
.
├── project/               # Source code
├── data/                  # HPatches dataset is in gitignore
├── figures/               # Figures and visualizations
├── report/                # LaTeX source and final report
└── README.md
```

---

## Report Structure

```text
1. Introduction

2. Dataset
    2.1 HPatches Benchmark
    2.2 Image Preparation

3. Methodology
    3.1 Overall Pipeline

    3.2 Feature Extraction
        3.2.1 SIFT
        3.2.2 ORB

    3.3 Feature Matching
        3.3.1 SIFT Matching
        3.3.2 ORB Matching
        3.3.3 Best Correspondence Selection

    3.4 Affine Transformation
        3.4.1 SIFT
        3.4.2 ORB

    3.5 Geometric Verification
        3.5.1 SIFT
        3.5.2 ORB

4. Results and Discussion

    4.1 Feature Detection Results
        - Number of Detected Keypoints
        - Detection Examples

    4.2 Feature Matching Results
        - Precision
        - Recall
        - Mean Matching Accuracy (MMA)

    4.3 Transformation and Robustness Results
        - Affine Transformation
        - Reprojection Error
        - Viewpoint Changes
        - Illumination Changes
        - Success and Failure Cases

    4.4 Performance Analysis
        - Runtime Comparison

    4.5 Discussion
```

---

## License

This repository was developed as part of the **EECS 4422: Computer Vision** course at **York University** during **Summer 2026**. It is intended for educational and academic purposes.
