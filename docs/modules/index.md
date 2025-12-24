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

[View Curriculum on GitHub](https://github.com/marcus888-techstack/introduction-to-opencv/tree/main/curriculum){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

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

| Module | Topics | README |
|:-------|:-------|:-------|
| **01: Core** | Arrays, pixel access, arithmetic, bitwise operations | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/01_core/README.md) |
| **02: Image Processing** | Filtering, morphology, edges, histograms, color spaces | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/02_imgproc/README.md) |
| **03: I/O & GUI** | Image/video I/O, windows, trackbars, drawing | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/03_io_gui/README.md) |
| **04: Features2D** | Corner detection, descriptors, feature matching | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/04_features2d/README.md) |
| **05: Object Detection** | Haar cascades, HOG, template matching | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/05_objdetect/README.md) |
| **06: Video Analysis** | Optical flow, background subtraction, tracking | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/06_video/README.md) |

---

## Advanced Modules (Week 3-4)

| Module | Topics | README |
|:-------|:-------|:-------|
| **07: Camera Calibration** | Intrinsics, distortion correction, perspective transform | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/07_calib3d/README.md) |
| **08: Deep Learning** | Model loading, blob preparation, inference | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/08_dnn/README.md) |
| **09: Machine Learning** | KNN, SVM, K-Means, Decision Trees | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/09_ml/README.md) |
| **10: Photo** | Inpainting, HDR, denoising, seamless cloning | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/10_photo/README.md) |
| **11: Stitching** | Panorama creation, homography, blending | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/11_stitching/README.md) |

---

## Extra Modules

| Module | Topics | README |
|:-------|:-------|:-------|
| **Face Recognition** | LBPH, EigenFace, FisherFace algorithms | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/extras/README.md) |
| **Object Tracking** | KCF, CSRT, MOSSE trackers | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/extras/README.md) |
| **OCR Integration** | Tesseract, EasyOCR integration | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/extras/README.md) |

---

## Key Functions Reference

| Module | Key Functions |
|:-------|:--------------|
| Core | `np.zeros`, `cv2.add`, `cv2.bitwise_and`, `cv2.split`, `cv2.merge` |
| ImgProc | `cv2.blur`, `cv2.Canny`, `cv2.findContours`, `cv2.threshold` |
| I/O & GUI | `cv2.imread`, `cv2.VideoCapture`, `cv2.createTrackbar` |
| Features2D | `cv2.ORB_create`, `cv2.SIFT_create`, `cv2.BFMatcher` |
| ObjDetect | `CascadeClassifier`, `cv2.matchTemplate`, `cv2.HOGDescriptor` |
| Video | `cv2.calcOpticalFlowPyrLK`, `cv2.createBackgroundSubtractorMOG2` |
| Calib3D | `cv2.calibrateCamera`, `cv2.undistort`, `cv2.warpPerspective` |
| DNN | `cv2.dnn.readNet`, `cv2.dnn.blobFromImage` |
| ML | `cv2.ml.KNearest_create`, `cv2.kmeans`, `cv2.ml.SVM_create` |
| Photo | `cv2.inpaint`, `cv2.seamlessClone`, `cv2.fastNlMeansDenoising` |
| Stitching | `cv2.Stitcher_create`, `cv2.detail.MultiBandBlender` |

---

## PDF Teaching Materials

Each module has a corresponding PDF guide for in-depth study:

[View All Teaching Materials]({{ site.baseurl }}/teaching-materials){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
