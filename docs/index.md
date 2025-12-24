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
| 1-2 | [Document Scanner]({{ site.baseurl }}/projects/01-document-scanner) | Edge detection, perspective transform, OCR |
| 1-2 | [Face Attendance]({{ site.baseurl }}/projects/02-face-attendance) | Face detection & recognition system |
| 3-4 | [License Plate Recognition]({{ site.baseurl }}/projects/03-license-plate) | ANPR for parking/security systems |
| 3-4 | [Object Counting]({{ site.baseurl }}/projects/04-object-counting) | People/vehicle tracking & analytics |
| 5-6 | [Quality Inspection]({{ site.baseurl }}/projects/05-quality-inspection) | Industrial defect detection |
| 5-6 | [Gesture Control]({{ site.baseurl }}/projects/06-gesture-control) | Touchless presentation control |

---

## Learning Modules

The curriculum covers all major OpenCV modules with **20+ hands-on tutorials**:

### Core Modules

| Module | Topics | Key Algorithms |
|:-------|:-------|:---------------|
| [Core]({{ site.baseurl }}/modules/01-core) | Arrays, operations, pixels | NumPy integration, bitwise ops |
| [Image Processing]({{ site.baseurl }}/modules/02-imgproc) | Filtering, morphology, edges | Gaussian blur, Canny, contours |
| [I/O & GUI]({{ site.baseurl }}/modules/03-io-gui) | Read/write, video, events | imread, VideoCapture, trackbars |
| [Features2D]({{ site.baseurl }}/modules/04-features2d) | Detection, matching | ORB, SIFT, FLANN, homography |
| [Object Detection]({{ site.baseurl }}/modules/05-objdetect) | Haar cascades, templates | Face detection, template matching |
| [Video Analysis]({{ site.baseurl }}/modules/06-video) | Optical flow, tracking | Lucas-Kanade, background subtraction |

### Advanced Modules

| Module | Topics | Key Algorithms |
|:-------|:-------|:---------------|
| [Camera Calibration]({{ site.baseurl }}/modules/07-calib3d) | 3D geometry, calibration | Undistortion, perspective transform |
| [Deep Learning]({{ site.baseurl }}/modules/08-dnn) | Neural networks | Model loading, blob, inference |
| [Machine Learning]({{ site.baseurl }}/modules/09-ml) | Traditional ML | KNN, SVM, K-Means |
| [Photo]({{ site.baseurl }}/modules/10-photo) | Enhancement | Inpainting, HDR, denoising |
| [Stitching]({{ site.baseurl }}/modules/11-stitching) | Panoramas | Feature alignment, blending |
| [Extras]({{ site.baseurl }}/modules/extras) | Face, tracking, OCR | LBPH, KCF, Tesseract |

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
