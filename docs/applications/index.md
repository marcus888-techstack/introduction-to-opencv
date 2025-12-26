---
layout: default
title: Applications
nav_order: 3
has_children: true
permalink: /applications
---

# Practical Applications
{: .fs-9 }

14 ready-to-run OpenCV applications demonstrating real-world computer vision techniques.
{: .fs-6 .fw-300 }

[View Applications on GitHub](https://github.com/marcus888-techstack/introduction-to-opencv/tree/main/curriculum/applications){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Application Overview

These standalone applications demonstrate OpenCV techniques in practical scenarios. Each application includes:
- Interactive mode with webcam support
- Demo mode with sample images
- Keyboard controls documented on startup

```
                    14 Practical Applications
┌────────────────────────────────────────────────────────────────┐
│                    BEGINNER (01-05)                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ Document │ │  Color   │ │ Realtime │ │  Face    │ │Object│ │
│  │ Scanner  │ │ Tracker  │ │ Filters  │ │  Blur    │ │Count │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────┘ │
├────────────────────────────────────────────────────────────────┤
│                  INTERMEDIATE (06-10)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ Motion   │ │   QR/    │ │  Lane    │ │  Image   │ │Color │ │
│  │  Alarm   │ │ Barcode  │ │ Detect   │ │Watermark │ │Palett│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────┘ │
├────────────────────────────────────────────────────────────────┤
│                    ADVANCED (11-14)                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐│
│  │    ArUco     │ │    Hand      │ │   Virtual    │ │Panoram││
│  │   Markers    │ │   Gesture    │ │  Background  │ │Stitch ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────┘│
└────────────────────────────────────────────────────────────────┘
```

---

## Beginner Applications

Perfect for learning fundamental OpenCV techniques.

| # | Application | Key Techniques | Use Case |
|:--|:------------|:---------------|:---------|
| 01 | [Document Scanner]({{ site.baseurl }}/applications/01-document-scanner) | Edge detection, Perspective transform | Mobile scanning apps |
| 02 | [Color Object Tracker]({{ site.baseurl }}/applications/02-color-tracker) | HSV color space, Contours | Robotics, games |
| 03 | [Real-time Filters]({{ site.baseurl }}/applications/03-realtime-filters) | Custom kernels, Blending | Instagram/TikTok |
| 04 | [Face Blur Privacy]({{ site.baseurl }}/applications/04-face-blur) | Cascade classifier, Blur | Privacy protection |
| 05 | [Object Counter]({{ site.baseurl }}/applications/05-object-counter) | Thresholding, Contours | Inventory counting |

---

## Intermediate Applications

Building on fundamentals with more sophisticated techniques.

| # | Application | Key Techniques | Use Case |
|:--|:------------|:---------------|:---------|
| 06 | [Motion Detection Alarm]({{ site.baseurl }}/applications/06-motion-alarm) | Background subtraction | Security cameras |
| 07 | [QR/Barcode Reader]({{ site.baseurl }}/applications/07-qr-barcode) | QRCodeDetector | Payments, inventory |
| 08 | [Lane Detection]({{ site.baseurl }}/applications/08-lane-detection) | Canny, Hough lines | Self-driving cars |
| 09 | [Image Watermarking]({{ site.baseurl }}/applications/09-watermark) | Alpha blending, LSB | Copyright protection |
| 10 | [Color Palette Extractor]({{ site.baseurl }}/applications/10-color-palette) | K-means clustering | Design tools |

---

## Advanced Applications

Combining multiple techniques for real-world solutions.

| # | Application | Key Techniques | Use Case |
|:--|:------------|:---------------|:---------|
| 11 | [ArUco Marker Detection]({{ site.baseurl }}/applications/11-aruco) | ArUco dictionary, Pose | Augmented reality |
| 12 | [Hand Gesture Recognition]({{ site.baseurl }}/applications/12-hand-gesture) | Skin segmentation, Hull | Gesture control |
| 13 | [Virtual Background]({{ site.baseurl }}/applications/13-virtual-background) | Background subtraction | Video conferencing |
| 14 | [Panorama Stitcher]({{ site.baseurl }}/applications/14-panorama) | Feature matching, Homography | Photography apps |

---

## Quick Start

```bash
# Download sample images first
python curriculum/sample_data/download_samples.py

# Run any application
python curriculum/applications/01_document_scanner.py
python curriculum/applications/07_qr_barcode_reader.py
python curriculum/applications/14_panorama_stitcher.py
```

---

## Techniques Matrix

| Application | ImgProc | Features | ObjDetect | Video | ML |
|:------------|:-------:|:--------:|:---------:|:-----:|:--:|
| Document Scanner | ✓ | | | | |
| Color Tracker | ✓ | | | ✓ | |
| Real-time Filters | ✓ | | | ✓ | |
| Face Blur | | | ✓ | ✓ | |
| Object Counter | ✓ | | | | |
| Motion Alarm | | | | ✓ | |
| QR Reader | | | ✓ | ✓ | |
| Lane Detection | ✓ | | | ✓ | |
| Watermarking | ✓ | | | | |
| Color Extractor | ✓ | | | | ✓ |
| ArUco Markers | | ✓ | ✓ | ✓ | |
| Hand Gesture | ✓ | | | ✓ | |
| Virtual Background | | | | ✓ | |
| Panorama | | ✓ | | | |

---

## Prerequisites

```bash
# Core OpenCV
pip install opencv-python numpy

# For advanced applications (ArUco, SIFT)
pip install opencv-contrib-python
```

---

## Application Structure

Each application follows a consistent pattern:

```python
# 1. Try real sample images first
img = get_image("sample.jpg")

# 2. Fall back to webcam if no sample
if img is None:
    cap = cv2.VideoCapture(0)

# 3. Fall back to synthetic demo if no webcam
if not cap.isOpened():
    demo_mode()
```

This ensures applications work in any environment while preferring real images for the best learning experience.
