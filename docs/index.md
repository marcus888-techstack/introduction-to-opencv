---
layout: default
title: Home
nav_order: 1
description: "A hands-on OpenCV course with 6 practical, real-world projects for intermediate Python students."
permalink: /
---

# Introduction to OpenCV
{: .fs-9 }

A hands-on OpenCV course with 6 practical, real-world projects for intermediate Python students.
{: .fs-6 .fw-300 }

[Get Started]({{ site.baseurl }}/getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/marcus888-techstack/introduction-to-opencv){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Course Overview

This course teaches computer vision through **hands-on projects**. Instead of just learning theory, you'll build real applications that solve practical problems.

```
Course Structure
================

    ┌─────────────────────────────────────────────────────────────────┐
    │                    FOUNDATIONS (Week 1)                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │    Core     │  │   ImgProc   │  │   I/O GUI   │             │
    │  │  Operations │  │  Filtering  │  │   Video     │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              FEATURES & DETECTION (Week 2)                      │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Features2D  │  │  ObjDetect  │  │    Video    │             │
    │  │  Matching   │  │    Haar     │  │   Tracking  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                ADVANCED TOPICS (Week 3)                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  Calib3D    │  │     DNN     │  │   ML/Photo  │             │
    │  │ Calibration │  │Deep Learning│  │  Stitching  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   PROJECTS (Week 4)                             │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
    │  │ DocScan │ │  Face   │ │ License │ │ Object  │ │ Gesture │  │
    │  │   OCR   │ │ Attend  │ │  Plate  │ │ Counter │ │ Control │  │
    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Practical Projects

Build these 6 real-world applications:

| Session | Project | What You'll Build |
|:--------|:--------|:------------------|
| 1-2 | [Document Scanner](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/01_document_scanner/README.md) | Edge detection, perspective transform, OCR |
| 1-2 | [Face Attendance](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/02_face_attendance/README.md) | Face detection & recognition system |
| 3-4 | [License Plate Recognition](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/03_license_plate/README.md) | ANPR for parking/security systems |
| 3-4 | [Object Counting](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/04_object_counting/README.md) | People/vehicle tracking & analytics |
| 5-6 | [Quality Inspection](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/05_quality_inspection/README.md) | Industrial defect detection |
| 5-6 | [Gesture Control](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/06_gesture_control/README.md) | Touchless presentation control |

---

## Learning Modules

The curriculum covers all major OpenCV modules with **20+ hands-on tutorials**:

### Core Modules

| Module | Topics | Key Algorithms |
|:-------|:-------|:---------------|
| [Core](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/01_core/README.md) | Arrays, operations, pixels | NumPy integration, bitwise ops |
| [Image Processing](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/02_imgproc/README.md) | Filtering, morphology, edges | Gaussian blur, Canny, contours |
| [I/O & GUI](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/03_io_gui/README.md) | Read/write, video, events | imread, VideoCapture, trackbars |
| [Features2D](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/04_features2d/README.md) | Detection, matching | ORB, SIFT, FLANN, homography |
| [Object Detection](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/05_objdetect/README.md) | Haar cascades, templates | Face detection, template matching |
| [Video Analysis](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/06_video/README.md) | Optical flow, tracking | Lucas-Kanade, background subtraction |

### Advanced Modules

| Module | Topics | Key Algorithms |
|:-------|:-------|:---------------|
| [Camera Calibration](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/07_calib3d/README.md) | 3D geometry, calibration | Undistortion, perspective transform |
| [Deep Learning](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/08_dnn/README.md) | Neural networks | Model loading, blob, inference |
| [Machine Learning](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/09_ml/README.md) | Traditional ML | KNN, SVM, K-Means |
| [Photo](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/10_photo/README.md) | Enhancement | Inpainting, HDR, denoising |
| [Stitching](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/11_stitching/README.md) | Panoramas | Feature alignment, blending |
| [Extras](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/extras/README.md) | Face, tracking, OCR | LBPH, KCF, Tesseract |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/marcus888-techstack/introduction-to-opencv.git
cd introduction-to-opencv

# Install dependencies
pip install -r requirements.txt

# Run a tutorial
python curriculum/01_core/01_basics.py

# Run a project
python projects/01_document_scanner/main.py
```

---

## Prerequisites

- **Python 3.8+** with intermediate proficiency
- **Webcam** for real-time projects
- Basic understanding of NumPy arrays

---

## Teaching Materials

Download comprehensive PDF guides for offline study:

[View All Teaching Materials]({{ site.baseurl }}/teaching-materials){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }

| Topic | PDF Guide |
|:------|:----------|
| Core & Image Processing | [Core Fundamentals]({{ site.baseurl }}/teaching_materials/01-core-image-fundamentals.pdf) |
| Features & Detection | [Feature Matching]({{ site.baseurl }}/teaching_materials/04-feature-matching.pdf) |
| Deep Learning | [DNN Inference]({{ site.baseurl }}/teaching_materials/08-deep-learning-inference.pdf) |
| Stitching | [Image Stitching]({{ site.baseurl }}/teaching_materials/11-image-stitching.pdf) |

---

## Official OpenCV Documentation

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)
- [Object Detection](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html)
- [Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html)
