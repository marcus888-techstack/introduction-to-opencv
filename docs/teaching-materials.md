---
layout: default
title: Teaching Materials
nav_order: 5
has_children: true
permalink: /teaching-materials
---

# Teaching Materials
{: .fs-9 }

Comprehensive PDF guides for each module with detailed explanations and visual diagrams.
{: .fs-6 .fw-300 }

---

## PDF Guides

These professionally formatted PDF documents provide in-depth coverage of each topic. Perfect for offline study or printing.

| Module | PDF Guide | Topics Covered |
|:-------|:----------|:---------------|
| **01: Core** | [Core Image Fundamentals]({{ site.baseurl }}/teaching_materials/01-core-image-fundamentals.pdf) | Image basics, NumPy arrays, pixel manipulation, ROI |
| **02: Image Processing** | [Practical Image Processing]({{ site.baseurl }}/teaching_materials/02-image-processing.pdf) | Filtering, morphology, thresholding, color spaces |
| **03: I/O & GUI** | [Interactive Framework]({{ site.baseurl }}/teaching_materials/03-io-gui-framework.pdf) | Image/video I/O, windows, trackbars, drawing |
| **04: Features2D** | [Feature Matching Pipeline]({{ site.baseurl }}/teaching_materials/04-feature-matching.pdf) | SIFT, ORB, feature matching, homography |
| **05: Object Detection** | [Object Detection Toolkit]({{ site.baseurl }}/teaching_materials/05-object-detection.pdf) | Haar cascades, HOG, template matching |
| **06: Video Analysis** | [Video Motion Analysis]({{ site.baseurl }}/teaching_materials/06-video-analysis.pdf) | Optical flow, background subtraction, tracking |
| **07: Machine Learning** | [Machine Learning]({{ site.baseurl }}/teaching_materials/07-machine-learning.pdf) | KNN, SVM, K-Means, Decision Trees |
| **08: Deep Learning** | [Deep Learning]({{ site.baseurl }}/teaching_materials/08-deep-learning.pdf) | Model loading, blob preparation, inference |
| **09: Multi-Object Tracking** | [Multi-Object Tracking]({{ site.baseurl }}/teaching_materials/09-multi-object-tracking.pdf) | MOT, MCMOT, person Re-ID |
| **10: 3D Vision** | [3D Vision]({{ site.baseurl }}/teaching_materials/10-3d-vision.pdf) | Calibration, stereo, 3D reconstruction, SfM |
| **11: Stitching** | [Image Stitching]({{ site.baseurl }}/teaching_materials/11-image-stitching.pdf) | Panoramas, homography, blending, projections |

---

## How to Use

1. **Download** - Click on any PDF link to download
2. **Study** - Each guide contains theory, algorithms, and code examples
3. **Practice** - Use alongside the tutorial files in the curriculum folder
4. **Reference** - Keep handy as a quick reference during development

---

## Recommended Reading Order

For beginners, we recommend following this sequence:

```
1. Core Image Fundamentals     → Understand image basics
2. Practical Image Processing  → Learn filtering & transforms
3. Interactive Framework       → Master I/O and GUI
4. Feature Matching Pipeline   → Explore feature detection
5. Object Detection Toolkit    → Build detection systems
6. Video Motion Analysis       → Work with video streams
7. OpenCV Vision Algorithms    → Apply classical ML
8. DNN From Model To Magic     → Use deep learning models
9. Multi-Camera Tracking       → MOT and Re-ID systems
10. 3D Vision Fundamentals     → Camera calibration & 3D
11. Image Stitching Mastered   → Create panoramas
```

---

## Need the Code?

Each PDF corresponds to a curriculum module with hands-on tutorial files:

```
curriculum/
├── 01_core/          → 01-core-image-fundamentals.pdf
├── 02_imgproc/       → 02-image-processing.pdf
├── 03_io_gui/        → 03-io-gui-framework.pdf
├── 04_features2d/    → 04-feature-matching.pdf
├── 05_objdetect/     → 05-object-detection.pdf
├── 06_video/         → 06-video-analysis.pdf
├── 07_ml/            → 07-machine-learning.pdf (4 tutorials)
├── 08_dnn/           → 08-deep-learning.pdf (3 tutorials)
├── 09_mcmot/         → 09-multi-object-tracking.pdf (5 tutorials)
├── 10_calib3d/       → 10-3d-vision.pdf (5 tutorials)
└── 11_stitching/     → 11-image-stitching.pdf (4 tutorials)
```

See the [Modules](/modules) section for detailed documentation of each module.
