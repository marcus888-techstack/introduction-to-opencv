---
layout: default
title: Modules
nav_order: 3
has_children: true
permalink: /modules
---

# Learning Modules
{: .fs-9 }

Comprehensive curriculum covering all major OpenCV modules with 20+ hands-on tutorials.
{: .fs-6 .fw-300 }

---

## Module Overview

```
                           OpenCV Curriculum
    ┌────────────────────────────────────────────────────────────┐
    │                     CORE MODULES                           │
    ├────────────────────────────────────────────────────────────┤
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
    │  │ 01 Core  │ │02 ImgProc│ │ 03 I/O   │ │04 Feature│      │
    │  │ Basics   │ │ Filters  │ │   GUI    │ │ Matching │      │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
    │  ┌──────────┐ ┌──────────┐                                │
    │  │05 Object │ │ 06 Video │                                │
    │  │  Detect  │ │ Analysis │                                │
    │  └──────────┘ └──────────┘                                │
    ├────────────────────────────────────────────────────────────┤
    │                   ADVANCED MODULES                         │
    ├────────────────────────────────────────────────────────────┤
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
    │  │07 Calib  │ │  08 DNN  │ │  09 ML   │ │ 10 Photo │      │
    │  │   3D     │ │DeepLearn │ │ Classic  │ │Computatnl│      │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
    │  ┌──────────┐ ┌──────────┐                                │
    │  │11 Stitch │ │  Extras  │                                │
    │  │ Panorama │ │Face/Track│                                │
    │  └──────────┘ └──────────┘                                │
    └────────────────────────────────────────────────────────────┘
```

---

## Core Modules (Week 1-2)

| Module | Topics | Tutorials |
|:-------|:-------|:----------|
| [01 Core]({{ site.baseurl }}/modules/01-core) | Arrays, pixel access, arithmetic, bitwise | 1 |
| [02 Image Processing]({{ site.baseurl }}/modules/02-imgproc) | Filtering, morphology, edges, histograms | 4 |
| [03 I/O & GUI]({{ site.baseurl }}/modules/03-io-gui) | Image/video I/O, windows, events | 3 |
| [04 Features2D]({{ site.baseurl }}/modules/04-features2d) | Corners, descriptors, matching | 3 |
| [05 Object Detection]({{ site.baseurl }}/modules/05-objdetect) | Haar cascades, template matching | 2 |
| [06 Video Analysis]({{ site.baseurl }}/modules/06-video) | Optical flow, background subtraction | 2 |

---

## Advanced Modules (Week 3-4)

| Module | Topics | Tutorials |
|:-------|:-------|:----------|
| [07 Camera Calibration]({{ site.baseurl }}/modules/07-calib3d) | Calibration, undistortion, perspective | 1 |
| [08 Deep Learning]({{ site.baseurl }}/modules/08-dnn) | Model loading, blob, inference | 1 |
| [09 Machine Learning]({{ site.baseurl }}/modules/09-ml) | KNN, SVM, K-Means, Decision Trees | 1 |
| [10 Photo]({{ site.baseurl }}/modules/10-photo) | Inpainting, HDR, denoising, cloning | 1 |
| [11 Stitching]({{ site.baseurl }}/modules/11-stitching) | Panorama creation, blending | 1 |
| [Extras]({{ site.baseurl }}/modules/extras) | Face recognition, tracking, OCR | 3 |

---

## Key Functions Summary

| Module | Key Functions |
|:-------|:--------------|
| Core | `np.zeros`, `cv2.add`, `cv2.bitwise_and` |
| ImgProc | `cv2.blur`, `cv2.Canny`, `cv2.findContours` |
| I/O | `cv2.imread`, `cv2.VideoCapture` |
| Features | `cv2.ORB_create`, `cv2.BFMatcher` |
| ObjDetect | `CascadeClassifier`, `cv2.matchTemplate` |
| Video | `cv2.calcOpticalFlowPyrLK`, `MOG2` |
| Calib3D | `cv2.calibrateCamera`, `cv2.warpPerspective` |
| DNN | `cv2.dnn.readNet`, `cv2.dnn.blobFromImage` |
| ML | `cv2.ml.KNearest_create`, `cv2.kmeans` |
| Photo | `cv2.inpaint`, `cv2.seamlessClone` |
| Stitching | `cv2.Stitcher_create` |
